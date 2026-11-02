#!/usr/bin/env python3
"""Forecast current node availability from Slurm's live scheduler view.

Reports:
  * fully idle nodes now, and the earliest node-specific pending job/reservation
    expected to use them;
  * partially free nodes now, and when their current allocations are expected
    to finish;
  * fully occupied nodes, and their expected release times.

Forecasts are estimates based on current job time limits and Slurm backfill
predictions. They are not guarantees that a node will remain free or be released
at the displayed time.
"""

from __future__ import annotations

import argparse
import datetime as dt
import shlex
import subprocess
import sys
from collections import defaultdict

UNKNOWN_TIMES = {"", "unknown", "n/a", "none", "null", "-", "(null)"}
UNAVAILABLE_FLAGS = {
    "DOWN", "DRAIN", "DRAINING", "FAIL", "FAILING", "NO_RESPOND",
    "NOT_RESPONDING", "MAINT", "POWERED_DOWN", "FUTURE",
}


def run_command(args: list[str], *, check: bool = True) -> str:
    proc = subprocess.run(args, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)
    if check and proc.returncode != 0:
        message = proc.stderr.strip() or f"command exited with status {proc.returncode}"
        raise RuntimeError(f"{' '.join(shlex.quote(x) for x in args)}: {message}")
    return proc.stdout if proc.returncode == 0 else ""


def parse_key_values(line: str) -> dict[str, str]:
    """Parse the whitespace-separated KEY=VALUE fields in scontrol -o output."""
    fields: dict[str, str] = {}
    for token in line.split():
        if "=" in token:
            key, value = token.split("=", 1)
            fields[key] = value
    return fields


def parse_slurm_datetime(value: str, now: dt.datetime) -> dt.datetime | None:
    value = value.strip()
    if value.lower() in UNKNOWN_TIMES:
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        # Fallback for common Slurm timestamp formats.
        parsed = None
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                parsed = dt.datetime.strptime(value, fmt)
                break
            except ValueError:
                pass
        if parsed is None:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=now.tzinfo)
    return parsed


def expand_nodelist(nodelist: str) -> list[str]:
    nodelist = nodelist.strip()
    if nodelist.lower() in UNKNOWN_TIMES:
        return []
    try:
        output = run_command(["scontrol", "show", "hostnames", nodelist])
        return [line.strip() for line in output.splitlines() if line.strip()]
    except RuntimeError:
        # A plain hostname may still be usable even if hostlist expansion fails.
        if "[" not in nodelist and "]" not in nodelist and "," not in nodelist:
            return [nodelist]
        return []


def parse_node_records(now: dt.datetime, partition_filter: str | None):
    text = run_command(["scontrol", "show", "nodes", "-o"])
    nodes: dict[str, dict] = {}
    excluded = 0

    for line in text.splitlines():
        f = parse_key_values(line)
        name = f.get("NodeName")
        if not name:
            continue
        partitions = f.get("Partitions", "")
        if partition_filter and partition_filter not in partitions.split(","):
            continue

        try:
            total = int(f.get("CPUTot", "0"))
            allocated = int(f.get("CPUAlloc", "0"))
        except ValueError:
            continue
        if total <= 0:
            continue

        state_raw = f.get("State", "UNKNOWN")
        state_flags = {part.upper() for part in state_raw.split("+")}
        if state_flags.intersection(UNAVAILABLE_FLAGS):
            excluded += 1
            continue

        free = max(0, total - allocated)
        nodes[name] = {
            "name": name,
            "state": state_raw,
            "total": total,
            "allocated": allocated,
            "free": free,
            "partitions": partitions or "?",
            "reservations": [],
        }

    return nodes, excluded


def get_running_jobs(now: dt.datetime):
    """Map each allocated node to the running/completing jobs using it."""
    fmt = "%i|%P|%u|%T|%e|%N"
    output = run_command(["squeue", "-h", "-t", "RUNNING,COMPLETING", "-o", fmt])
    jobs_by_node: dict[str, list[dict]] = defaultdict(list)

    for line in output.splitlines():
        parts = line.split("|", 5)
        if len(parts) != 6:
            continue
        job_id, partition, user, state, end_text, nodelist = parts
        end_time = parse_slurm_datetime(end_text, now)
        for node in expand_nodelist(nodelist):
            jobs_by_node[node].append({
                "id": job_id,
                "partition": partition,
                "user": user,
                "state": state,
                "end": end_time,
                "end_text": end_text,
            })

    return jobs_by_node


def get_pending_node_claims(now: dt.datetime):
    """Return backfill's node-specific pending-job start estimates."""
    fmt = "%i|%P|%u|%S|%Y|%r"
    output = run_command(["squeue", "--start", "-h", "-t", "PENDING", "-o", fmt], check=False)
    claims: dict[str, list[dict]] = defaultdict(list)
    unplaced = 0

    for line in output.splitlines():
        parts = line.split("|", 5)
        if len(parts) != 6:
            continue
        job_id, partition, user, start_text, expected_nodes, reason = parts
        start_time = parse_slurm_datetime(start_text, now)
        node_names = expand_nodelist(expected_nodes)
        if start_time is None or not node_names:
            unplaced += 1
            continue
        for node in node_names:
            claims[node].append({
                "id": job_id,
                "partition": partition,
                "user": user,
                "start": start_time,
                "start_text": start_text,
                "reason": reason,
            })

    for node in claims:
        claims[node].sort(key=lambda item: item["start"])
    return claims, unplaced


def get_reservations(now: dt.datetime):
    """Return current/future reservations by node; gracefully tolerate access limits."""
    output = run_command(["scontrol", "show", "reservation", "-o"], check=False)
    reservations: dict[str, list[dict]] = defaultdict(list)

    for line in output.splitlines():
        f = parse_key_values(line)
        name = f.get("ReservationName")
        if not name:
            continue
        start = parse_slurm_datetime(f.get("StartTime", ""), now)
        end = parse_slurm_datetime(f.get("EndTime", ""), now)
        if not start or not end or end <= now:
            continue
        node_names = expand_nodelist(f.get("Nodes", ""))
        for node in node_names:
            reservations[node].append({"name": name, "start": start, "end": end})

    for node in reservations:
        reservations[node].sort(key=lambda item: item["start"])
    return reservations


def fmt_time(value: dt.datetime | None) -> str:
    if value is None:
        return "UNKNOWN"
    return value.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def fmt_delta(value: dt.datetime | None, now: dt.datetime) -> str:
    if value is None:
        return "duration unknown"
    seconds = max(0, int((value - now).total_seconds()))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, _ = divmod(rem, 60)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours or days:
        parts.append(f"{hours}h")
    parts.append(f"{minutes}m")
    return "in " + " ".join(parts)


def next_claim_for_node(node: str, pending: dict, reservations: dict, now: dt.datetime):
    events = []
    for job in pending.get(node, []):
        if job["start"] >= now:
            events.append((job["start"], "pending job", job["id"]))
    for reservation in reservations.get(node, []):
        if reservation["start"] > now:
            events.append((reservation["start"], "reservation", reservation["name"]))
        elif reservation["start"] <= now < reservation["end"]:
            # Active reservation: the node may be physically idle but reserved.
            events.append((now, "active reservation until", reservation["name"], reservation["end"]))
    if not events:
        return None
    events.sort(key=lambda x: x[0])
    return events[0]


def node_release_time(node: dict, jobs_by_node: dict):
    jobs = jobs_by_node.get(node["name"], [])
    if not jobs:
        return None, []
    if any(job["end"] is None for job in jobs):
        return None, jobs
    return max(job["end"] for job in jobs), jobs


def print_claim(node_name: str, pending: dict, reservations: dict, now: dt.datetime):
    event = next_claim_for_node(node_name, pending, reservations, now)
    if event is None:
        print("    Next scheduled use: no node-specific future claim reported")
        return
    when, kind, identifier, *extra = event
    if kind == "active reservation until":
        end = extra[0] if extra else None
        print(f"    Reservation active: {identifier}; reserved until {fmt_time(end)}")
    else:
        print(f"    Earliest forecast claim: {fmt_time(when)} ({fmt_delta(when, now)})")
        print(f"    Claim source: {kind} {identifier}")


def main():
    parser = argparse.ArgumentParser(
        description="Show current Slurm node availability and estimated release/claim times."
    )
    parser.add_argument("--partition", help="Only show nodes in this partition (default: all partitions)")
    args = parser.parse_args()

    now = dt.datetime.now().astimezone()
    try:
        nodes, excluded = parse_node_records(now, args.partition)
        jobs_by_node = get_running_jobs(now)
        pending, unplaced_pending = get_pending_node_claims(now)
        reservations = get_reservations(now)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    empty, partial, occupied = [], [], []
    for node in nodes.values():
        if node["free"] == node["total"]:
            empty.append(node)
        elif node["free"] > 0:
            partial.append(node)
        else:
            occupied.append(node)

    sort_key = lambda n: (n["free"] * -1, n["name"])
    empty.sort(key=lambda n: n["name"])
    partial.sort(key=sort_key)
    occupied.sort(key=lambda n: n["name"])

    print("\n================ SLURM NODE FORECAST ================")
    print(f"Checked at : {fmt_time(now)}")
    print(f"Scope      : partition {args.partition}" if args.partition else "Scope      : all partitions (unique physical nodes)")
    print(f"Nodes seen : {len(nodes)} | excluded unavailable nodes: {excluded}")
    print(f"Pending jobs without a usable node-specific start prediction: {unplaced_pending}")

    print(f"\n[1] FULLY EMPTY NOW: {len(empty)} node(s)")
    if not empty:
        print("  None")
    for n in empty:
        print(f"  {n['name']:<14} {n['free']:>3}/{n['total']:<3} CPUs free | state={n['state']} | partitions={n['partitions']}")
        print_claim(n["name"], pending, reservations, now)

    print(f"\n[2] PARTIALLY FREE NOW: {len(partial)} node(s)")
    if not partial:
        print("  None")
    for n in partial:
        release, jobs = node_release_time(n, jobs_by_node)
        print(f"  {n['name']:<14} {n['free']:>3}/{n['total']:<3} CPUs free, {n['allocated']:>3} allocated | state={n['state']} | partitions={n['partitions']}")
        if release is None:
            print("    Fully-free estimate: UNKNOWN (could not map every current allocation to a job end time)")
        else:
            print(f"    Fully-free estimate: {fmt_time(release)} ({fmt_delta(release, now)})")
        for job in jobs:
            print(f"    Current job {job['id']} ({job['user']}, {job['partition']}, {job['state']}): ends {fmt_time(job['end'])}")
        print_claim(n["name"], pending, reservations, now)

    print(f"\n[3] FULLY OCCUPIED NOW: {len(occupied)} node(s)")
    if not occupied:
        print("  None")
    for n in occupied:
        release, jobs = node_release_time(n, jobs_by_node)
        print(f"  {n['name']:<14} {n['free']:>3}/{n['total']:<3} CPUs free | state={n['state']} | partitions={n['partitions']}")
        if release is None:
            print("    Fully-free estimate: UNKNOWN (could not map every current allocation to a job end time)")
        else:
            print(f"    Fully-free estimate: {fmt_time(release)} ({fmt_delta(release, now)})")
        for job in jobs:
            print(f"    Current job {job['id']} ({job['user']}, {job['partition']}, {job['state']}): ends {fmt_time(job['end'])}")
        print_claim(n["name"], pending, reservations, now)

    print("\n[4] SUMMARY")
    print(f"  Fully empty now       : {len(empty)}")
    print(f"  Partially free now    : {len(partial)}")
    print(f"  Fully occupied now    : {len(occupied)}")
    print(f"  Total eligible nodes  : {len(empty) + len(partial) + len(occupied)}")
    print(f"  Free CPUs right now   : {sum(n['free'] for n in nodes.values())}")
    print("\nNotes:")
    print("  * Release estimates use the scheduled end times of jobs currently on each node.")
    print("  * Empty-node claim estimates use squeue --start's expected node list (%Y), plus reservations.")
    print("  * Slurm start/end predictions can change when jobs are submitted, resized, requeued, cancelled, or extended.")
    print("  * A node becoming idle does not guarantee that your job/QoS/partition can obtain it.")
    print("  * 'CPUs' are Slurm CPU units; their mapping to physical cores/hardware threads depends on cluster configuration.")
    print("======================================================\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
