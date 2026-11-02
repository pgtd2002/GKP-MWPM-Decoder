#!/usr/bin/env python3
"""Slurm CPU-chunk availability and forecast.

A CPU chunk is defined as N Slurm CPU units on the SAME node. For this
cluster's common 48-CPU nodes and a job using --cpus-per-task=48 with
--ntasks-per-node=1, use --chunk-size 48.

The script reports:
  * CPU chunks available on nodes right now and the earliest predicted claim;
  * future chunks expected when currently running jobs finish;
  * an optional forecast for when a requested number of chunks may be
    simultaneously available.

Forecasts use squeue --start's backfill estimate and current job time limits.
They are estimates, not guarantees. A blank/unknown expected node list means
Slurm has not supplied a node-specific start prediction for that job.
"""

import argparse
import datetime as dt
import shlex
import subprocess
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

UNKNOWN = {"", "unknown", "n/a", "none", "null", "-", "(null)", "(none)", "unassigned"}
UNAVAILABLE_FLAGS = {
    "DOWN", "DRAIN", "DRAINING", "FAIL", "FAILING", "NO_RESPOND",
    "NOT_RESPONDING", "MAINT", "POWERED_DOWN", "FUTURE", "INVALID_REG",
}


def run_command(args: List[str], check: bool = True) -> str:
    proc = subprocess.run(args, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)
    if check and proc.returncode != 0:
        message = proc.stderr.strip() or "command exited with status {}".format(proc.returncode)
        raise RuntimeError("{}: {}".format(" ".join(shlex.quote(x) for x in args), message))
    return proc.stdout if proc.returncode == 0 else ""


def parse_key_values(line: str) -> Dict[str, str]:
    result = {}
    for token in line.split():
        if "=" in token:
            key, value = token.split("=", 1)
            result[key] = value
    return result


def parse_datetime(value: str, now: dt.datetime) -> Optional[dt.datetime]:
    value = (value or "").strip()
    if value.lower() in UNKNOWN:
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        parsed = None
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M"):
            try:
                parsed = dt.datetime.strptime(value, fmt)
                break
            except ValueError:
                continue
        if parsed is None:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=now.tzinfo)
    return parsed.astimezone(now.tzinfo)


def expand_nodelist(nodelist: str) -> List[str]:
    nodelist = (nodelist or "").strip()
    if nodelist.lower() in UNKNOWN or nodelist.upper() in {"ALL", "NO_VAL"}:
        return []
    try:
        out = run_command(["scontrol", "show", "hostnames", nodelist])
        return [line.strip() for line in out.splitlines() if line.strip()]
    except RuntimeError:
        if not any(ch in nodelist for ch in "[],"):
            return [nodelist]
        return []


def read_nodes(partition: Optional[str]):
    output = run_command(["scontrol", "show", "nodes", "-o"])
    nodes = {}
    excluded = 0
    for line in output.splitlines():
        fields = parse_key_values(line)
        name = fields.get("NodeName")
        if not name:
            continue
        partitions = fields.get("Partitions", "")
        if partition and partition not in partitions.split(","):
            continue
        try:
            total = int(fields.get("CPUTot", "0"))
            allocated = int(fields.get("CPUAlloc", "0"))
        except ValueError:
            continue
        if total <= 0:
            continue
        state = fields.get("State", "UNKNOWN")
        state_flags = {piece.upper() for piece in state.split("+")}
        if state_flags.intersection(UNAVAILABLE_FLAGS):
            excluded += 1
            continue
        free = max(0, total - allocated)
        nodes[name] = {
            "name": name,
            "state": state,
            "total": total,
            "allocated": allocated,
            "free": free,
            "partitions": partitions or "?",
        }
    return nodes, excluded


def read_running_jobs(now: dt.datetime):
    # %e is the expected end time based on the current job time limit.
    fmt = "%i|%P|%u|%T|%e|%N"
    output = run_command(["squeue", "-h", "-t", "RUNNING,COMPLETING", "-o", fmt])
    by_node = defaultdict(list)
    for line in output.splitlines():
        parts = line.split("|", 5)
        if len(parts) != 6:
            continue
        job_id, partition, user, state, end_text, node_list = parts
        end_time = parse_datetime(end_text, now)
        for node in expand_nodelist(node_list):
            by_node[node].append({
                "id": job_id, "partition": partition, "user": user,
                "state": state, "end": end_time, "end_text": end_text,
            })
    return by_node


def read_pending_claims(now: dt.datetime):
    # %S = expected start; %Y = expected node list for pending jobs.
    fmt = "%i|%P|%u|%S|%Y|%r"
    output = run_command(["squeue", "--start", "-h", "-t", "PENDING", "-o", fmt], check=False)
    by_node = defaultdict(list)
    no_node_prediction = 0
    for line in output.splitlines():
        parts = line.split("|", 5)
        if len(parts) != 6:
            continue
        job_id, partition, user, start_text, expected_nodes, reason = parts
        start = parse_datetime(start_text, now)
        node_names = expand_nodelist(expected_nodes)
        if start is None or not node_names:
            no_node_prediction += 1
            continue
        for node in node_names:
            by_node[node].append({
                "id": job_id, "partition": partition, "user": user,
                "start": start, "reason": reason,
            })
    for node in by_node:
        by_node[node].sort(key=lambda item: item["start"])
    return by_node, no_node_prediction


def read_reservations(now: dt.datetime):
    output = run_command(["scontrol", "show", "reservation", "-o"], check=False)
    by_node = defaultdict(list)
    for line in output.splitlines():
        fields = parse_key_values(line)
        name = fields.get("ReservationName")
        if not name:
            continue
        start = parse_datetime(fields.get("StartTime", ""), now)
        end = parse_datetime(fields.get("EndTime", ""), now)
        if start is None or end is None or end <= now:
            continue
        for node in expand_nodelist(fields.get("Nodes", "")):
            by_node[node].append({"name": name, "start": start, "end": end})
    for node in by_node:
        by_node[node].sort(key=lambda item: item["start"])
    return by_node


def release_time(node: dict, jobs_by_node: dict, now: dt.datetime):
    """When the node is expected to become entirely unallocated."""
    if node["allocated"] == 0:
        return now, []
    jobs = jobs_by_node.get(node["name"], [])
    if not jobs or any(job["end"] is None for job in jobs):
        return None, jobs
    return max(job["end"] for job in jobs), jobs


def active_reservations(node_name: str, reservations: dict, when: dt.datetime):
    return [r for r in reservations.get(node_name, []) if r["start"] <= when < r["end"]]


def next_claim(node_name: str, pending: dict, reservations: dict,
               after: dt.datetime) -> Optional[Tuple[dt.datetime, str, str]]:
    events = []
    for job in pending.get(node_name, []):
        if job["start"] >= after:
            events.append((job["start"], "pending job", job["id"]))
    for reservation in reservations.get(node_name, []):
        if reservation["start"] >= after:
            events.append((reservation["start"], "reservation", reservation["name"]))
    if not events:
        return None
    return min(events, key=lambda item: item[0])


def next_available_start(node_name: str, start: dt.datetime, reservations: dict):
    """Move a prospective availability start past any overlapping reservation."""
    moved = start
    # A few passes handle back-to-back/overlapping reservations.
    for _ in range(10):
        overlapping = active_reservations(node_name, reservations, moved)
        if not overlapping:
            break
        new_moved = max(r["end"] for r in overlapping)
        if new_moved <= moved:
            break
        moved = new_moved
    return moved


def fmt_time(value: Optional[dt.datetime]) -> str:
    if value is None:
        return "UNKNOWN"
    return value.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def fmt_duration(start: dt.datetime, end: Optional[dt.datetime]) -> str:
    if end is None:
        return "no node-specific expiry predicted"
    seconds = int((end - start).total_seconds())
    if seconds <= 0:
        return "no positive window predicted"
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, _ = divmod(rem, 60)
    bits = []
    if days:
        bits.append("{}d".format(days))
    if hours or days:
        bits.append("{}h".format(hours))
    bits.append("{}m".format(minutes))
    return " ".join(bits)


def node_claim_text(node_name: str, pending: dict, reservations: dict, now: dt.datetime):
    active = active_reservations(node_name, reservations, now)
    if active:
        end = max(r["end"] for r in active)
        names = ",".join(r["name"] for r in active)
        return "RESERVED now ({}) until {}".format(names, fmt_time(end)), end
    event = next_claim(node_name, pending, reservations, now)
    if event is None:
        return "no node-specific claim reported", None
    when, kind, identifier = event
    return "{} {} at {}".format(kind, identifier, fmt_time(when)), when


def build_node_forecasts(nodes: dict, jobs_by_node: dict, pending: dict,
                         reservations: dict, now: dt.datetime, chunk_size: int):
    forecasts = {}
    for name, node in nodes.items():
        total_chunks = node["total"] // chunk_size
        current_chunks_raw = node["free"] // chunk_size
        active_res = active_reservations(name, reservations, now)
        current_chunks = 0 if active_res else current_chunks_raw
        release, jobs = release_time(node, jobs_by_node, now)
        claim_now = next_claim(name, pending, reservations, now)
        claim_now_time = claim_now[0] if claim_now else None

        future_start = None
        future_end = None
        future_chunks = 0
        future_claim = None
        future_note = ""

        if release is not None and (release > now or active_res) and total_chunks > current_chunks:
            base_start = release
            if active_res:
                base_start = max(base_start, max(r["end"] for r in active_res))
            base_start = next_available_start(name, base_start, reservations)
            candidate = next_claim(name, pending, reservations, base_start)

            # If a pending claim is predicted before the jobs release the node,
            # don't incorrectly promise a later free window from this snapshot.
            if ((claim_now_time is not None and claim_now_time < base_start)
                    or (candidate is not None and candidate[0] <= base_start)):
                future_note = "a node-specific claim/reservation is predicted before or at the availability time"
            else:
                future_start = base_start
                future_claim = candidate
                future_end = candidate[0] if candidate else None
                future_chunks = max(0, total_chunks - current_chunks)

        forecasts[name] = {
            "node": node,
            "total_chunks": total_chunks,
            "current_chunks": current_chunks,
            "raw_current_chunks": current_chunks_raw,
            "active_reservations": active_res,
            "release": release,
            "jobs": jobs,
            "claim_now": claim_now,
            "claim_now_time": claim_now_time,
            "future_start": future_start,
            "future_end": future_end,
            "future_claim": future_claim,
            "future_chunks": future_chunks,
            "future_note": future_note,
        }
    return forecasts


def target_window(forecasts: dict, target: int, now: dt.datetime):
    """Find the first forecast interval with at least target same-node CPU chunks."""
    count_now = sum(f["current_chunks"] for f in forecasts.values())
    events = defaultdict(int)

    for f in forecasts.values():
        current = f["current_chunks"]
        future_start = f["future_start"]

        if future_start is not None:
            # Current chunks continue through the release time; the future
            # increment adds the additional chunks released by the node.
            events[future_start] += f["future_chunks"]
            if f["future_end"] is not None:
                # At the next predicted claim, all chunks on this node are
                # treated as unavailable (a conservative whole-node model).
                events[f["future_end"]] -= f["total_chunks"]
        elif current > 0 and f["claim_now_time"] is not None and f["claim_now_time"] > now:
            # No later release window was safely identified, so only count
            # the capacity that exists now, until its next predicted claim.
            events[f["claim_now_time"]] -= current

    count = count_now
    times = sorted(t for t in events if t > now)

    if count >= target:
        for t in times:
            count += events[t]
            if count < target:
                return now, t, count_now
        return now, None, count_now

    for i, t in enumerate(times):
        count += events[t]
        if count >= target:
            end = None
            for later in times[i + 1:]:
                count += events[later]
                if count < target:
                    end = later
                    break
            return t, end, count_now

    return None, None, count_now


def main():
    parser = argparse.ArgumentParser(
        description="Report node-level free CPU chunks now and forecast future availability."
    )
    parser.add_argument("--partition", help="Limit the scan to one partition, e.g. large or medium")
    parser.add_argument("--chunk-size", type=int, default=48,
                        help="CPU units required per chunk ON ONE NODE (default: 48)")
    parser.add_argument("--target-chunks", type=int, default=0,
                        help="Optional: forecast when this many chunks of --chunk-size CPUs may be simultaneously available")
    parser.add_argument("--hide-occupied", action="store_true",
                        help="Hide the detailed list of fully occupied nodes")
    args = parser.parse_args()

    if args.chunk_size <= 0:
        parser.error("--chunk-size must be a positive integer")
    if args.target_chunks < 0:
        parser.error("--target-chunks cannot be negative")

    now = dt.datetime.now().astimezone()
    try:
        nodes, excluded = read_nodes(args.partition)
        jobs_by_node = read_running_jobs(now)
        pending, no_node_prediction = read_pending_claims(now)
        reservations = read_reservations(now)
    except RuntimeError as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        return 2

    forecasts = build_node_forecasts(nodes, jobs_by_node, pending, reservations, now, args.chunk_size)

    free_nodes = []
    partial_nodes = []
    reserved_nodes = []
    occupied_nodes = []
    for f in forecasts.values():
        node = f["node"]
        if f["active_reservations"]:
            if node["free"] > 0:
                reserved_nodes.append(f)
            else:
                occupied_nodes.append(f)
        elif node["free"] >= args.chunk_size:
            free_nodes.append(f)
        elif node["free"] > 0:
            partial_nodes.append(f)
        else:
            occupied_nodes.append(f)

    free_nodes.sort(key=lambda f: (-f["current_chunks"], f["node"]["name"]))
    partial_nodes.sort(key=lambda f: (-f["node"]["free"], f["node"]["name"]))
    reserved_nodes.sort(key=lambda f: f["node"]["name"])
    future_nodes = [f for f in forecasts.values() if f["future_chunks"] > 0]
    future_nodes.sort(key=lambda f: (f["future_start"], f["node"]["name"]))
    blocked_future_nodes = [f for f in forecasts.values() if f["future_note"]]
    blocked_future_nodes.sort(key=lambda f: f["node"]["name"])

    total_raw_free = sum(f["node"]["free"] for f in forecasts.values())
    current_chunks = sum(f["current_chunks"] for f in forecasts.values())
    current_chunk_cpus = current_chunks * args.chunk_size
    future_chunks = sum(f["future_chunks"] for f in future_nodes)

    print("\n================ SLURM CPU-CHUNK FORECAST ================")
    print("Checked at       : {}".format(fmt_time(now)))
    print("Scope            : {}".format("partition {}".format(args.partition) if args.partition else "all partitions (unique nodes)"))
    print("Chunk definition : {} Slurm CPUs on the SAME node".format(args.chunk_size))
    print("Nodes considered : {} | unavailable nodes excluded: {}".format(len(nodes), excluded))
    print("Pending jobs without node-specific start prediction: {}".format(no_node_prediction))
    print("\n[1] CHUNKS AVAILABLE NOW")
    print("Node             Free CPUs   Full chunks    Available until / forecast claim")
    print("-------------------------------------------------------------------------------")
    if not free_nodes:
        print("No unreserved node currently has at least one full CPU chunk.")
    for f in free_nodes:
        node = f["node"]
        claim_text, claim_time = node_claim_text(node["name"], pending, reservations, now)
        duration = fmt_duration(now, claim_time)
        print("{:<16} {:>3}/{:<3}     {:>3} x {:<2}      {} ({})".format(
            node["name"], node["free"], node["total"], f["current_chunks"], args.chunk_size,
            claim_text, duration))
        if node["free"] % args.chunk_size:
            print("  residual on node: {} CPU(s) below one complete chunk".format(node["free"] % args.chunk_size))

    print("\n[2] PARTIAL CPU FRAGMENTS (not enough for one full chunk on a node)")
    if not partial_nodes:
        print("None")
    for f in partial_nodes:
        node = f["node"]
        claim_text, claim_time = node_claim_text(node["name"], pending, reservations, now)
        print("  {:<14} {:>3}/{:<3} free CPUs | {} ({})".format(
            node["name"], node["free"], node["total"], claim_text, fmt_duration(now, claim_time)))

    print("\n[3] PHYSICALLY FREE BUT RESERVED NOW")
    if not reserved_nodes:
        print("None")
    for f in reserved_nodes:
        node = f["node"]
        claim_text, _ = node_claim_text(node["name"], pending, reservations, now)
        print("  {:<14} {:>3}/{:<3} free CPUs | {}".format(
            node["name"], node["free"], node["total"], claim_text))

    print("\n[4] FUTURE FULL-NODE CAPACITY (after current jobs release CPUs)")
    print("Node             Currently free   New chunks   Available from                 Expected until")
    print("-----------------------------------------------------------------------------------------------")
    if not future_nodes:
        print("No future full-node CPU chunk window could be predicted from the current Slurm snapshot.")
    for f in future_nodes:
        node = f["node"]
        if f["future_claim"]:
            until_text = "{} ({})".format(fmt_time(f["future_end"]), f["future_claim"][1] + " " + f["future_claim"][2])
            duration = fmt_duration(f["future_start"], f["future_end"])
        else:
            until_text = "No node-specific expiry predicted"
            duration = "unknown horizon"
        print("{:<16} {:>3}/{:<3}        {:>3}          {:<29} {} [{}]".format(
            node["name"], node["free"], node["total"], f["future_chunks"],
            fmt_time(f["future_start"]), until_text, duration))
        if f["future_note"]:
            print("  NOTE: {}".format(f["future_note"]))
        for job in f["jobs"]:
            print("  Current job {} ({}, {}, {}): ends {}".format(
                job["id"], job["user"], job["partition"], job["state"], fmt_time(job["end"])))

    if blocked_future_nodes:
        print("\n[5] NODES WITH NO SAFE FUTURE CHUNK WINDOW IN THIS SNAPSHOT")
        for f in blocked_future_nodes:
            node = f["node"]
            claim_text, _ = node_claim_text(node["name"], pending, reservations, now)
            print("  {:<14} full-release estimate: {} | {}".format(
                node["name"], fmt_time(f["release"]), f["future_note"]))
            print("    Next reported event: {}".format(claim_text))

    if not args.hide_occupied:
        print("\n[6] FULLY OCCUPIED NODES")
        if not occupied_nodes:
            print("None")
        for f in sorted(occupied_nodes, key=lambda x: x["node"]["name"]):
            node = f["node"]
            release = f["release"]
            jobs = f["jobs"]
            if release is None:
                print("  {}: {}/{} allocated; full-release time UNKNOWN".format(
                    node["name"], node["allocated"], node["total"]))
            else:
                print("  {}: {}/{} allocated; full release {} ({})".format(
                    node["name"], node["allocated"], node["total"], fmt_time(release), fmt_duration(now, release)))
            for job in jobs:
                print("    Job {} ({}, {}, {}): ends {}".format(
                    job["id"], job["user"], job["partition"], job["state"], fmt_time(job["end"])))

    print("\n[7] SUMMARY" if not args.hide_occupied or blocked_future_nodes else "\n[6] SUMMARY")
    print("  Nodes with >=1 full chunk now : {}".format(len(free_nodes)))
    print("  Full CPU chunks available now : {} x {} = {} Slurm CPUs".format(
        current_chunks, args.chunk_size, current_chunk_cpus))
    print("  Raw free Slurm CPUs now       : {} (includes partial fragments/reserved nodes)".format(total_raw_free))
    print("  Nodes with partial fragments  : {}".format(len(partial_nodes)))
    print("  Reserved nodes with free CPUs : {}".format(len(reserved_nodes)))
    print("  Future full-node chunks       : {} (across {} nodes)".format(future_chunks, len(future_nodes)))
    print("  Releases without safe window  : {}".format(len(blocked_future_nodes)))

    if args.target_chunks:
        start, end, now_count = target_window(forecasts, args.target_chunks, now)
        print("\n  TARGET REQUEST: {} simultaneous chunk(s) of {} CPUs each".format(
            args.target_chunks, args.chunk_size))
        print("  Current qualifying chunks      : {}".format(now_count))
        if start is None:
            print("  Forecast                        : target not reached by current node/job predictions")
        else:
            print("  Earliest forecast start         : {} ({})".format(fmt_time(start), fmt_duration(now, start)))
            if end is None:
                print("  Forecast window                 : no predicted end from current node-specific claims")
            else:
                print("  Forecast window ends             : {} (window {})".format(fmt_time(end), fmt_duration(start, end)))

    print("\nNotes:")
    print("  * A chunk means --chunk-size CPUs on one node; fragments on different nodes are NOT combined.")
    print("  * For your 48-worker, one-task-per-node setup, --chunk-size 48 is the relevant setting.")
    print("  * A full-release time is based on the latest expected end time of jobs currently mapped to that node.")
    print("  * A node-specific pending claim uses squeue --start (%S and %Y); this requires Slurm backfill predictions.")
    print("  * No reported claim does NOT guarantee that a node will stay free; a newly submitted job can take it.")
    print("  * Partition/QOS/account limits, reservations, memory and job priority can still block your allocation.")
    print("  * CPU counts are Slurm CPU units; their mapping to physical cores/hardware threads is cluster-config dependent.")
    print("===========================================================\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
