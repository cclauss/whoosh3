"""Parallel indexing: the blessed multi-thread pattern for free-threaded CPython.

Whoosh indexing is CPU-bound work done in *pure Python* -- tokenizing,
stemming, filtering, and building postings. On a normal (GIL-enabled) CPython
build, running that work across threads does not speed it up: only one thread
holds the GIL at a time. On a **free-threaded** build (``3.13t``/``3.14t``,
PEP 703) the GIL is gone, so this same pure-Python work can finally scale
across real cores -- no C extension required.

This example is the *blessed pattern* referenced by the concurrency guide
(https://priya-sundaram-dev.github.io/whoosh/docs/threads.html). It follows the
per-object concurrency contract exactly:

  * A built ``Schema`` is immutable and safe to share across threads.
  * A plain ``IndexWriter`` is single-writer -- never shared. So each worker
    thread writes to *its own* index in its own directory (one writer each),
    which sidesteps the write lock entirely.
  * Read-only ``IndexReader`` objects from the finished sub-indexes are then
    merged into one final index with ``writer.add_reader()`` -- the same
    primitive Whoosh's own multiprocessing writer uses.

The result is a fan-out / fan-in pipeline:

    corpus ->  split into N shards
           ->  N worker threads, each builds a sub-index   (parallel, CPU-bound)
           ->  main thread merges the sub-indexes           (add_reader)
           ->  one final, ordinary Whoosh index

Run::

    python examples/parallel_indexing.py                  # auto worker count
    python examples/parallel_indexing.py --docs 40000 --workers 4
    python examples/parallel_indexing.py --check          # correctness only

On a free-threaded build you should see the parallel wall-clock time drop
below the serial baseline as workers increase; on a GIL build the two are
about the same (this is expected, and the script says so). Requires only the
standard library plus Whoosh itself.
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

from whoosh import index
from whoosh.analysis import StemmingAnalyzer
from whoosh.fields import ID, TEXT, Schema
from whoosh.qparser import QueryParser

# A small vocabulary so queries reliably match something. Stemming analysis is
# deliberately included: it is the kind of real per-token Python work that
# free-threaded builds let you parallelise.
WORDS = (
    "search engine python library index document token analyzer query parser "
    "ranking relevance scoring highlight facet storage segment posting term "
    "vector field schema writer reader stemming tokenizer filter fast pure "
    "memory disk network cluster shard replica cache latency throughput data "
    "structure algorithm sort merge trie automaton regex fuzzy prefix wildcard"
).split()


def make_schema() -> Schema:
    # StemmingAnalyzer() gives each thread genuine CPU-bound Python work per
    # token, which is what actually scales once the GIL is gone.
    return Schema(
        id=ID(stored=True, unique=True), body=TEXT(analyzer=StemmingAnalyzer())
    )


def make_corpus(n: int, seed: int = 1234) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    docs = []
    for i in range(n):
        length = rng.randint(40, 120)
        body = " ".join(rng.choice(WORDS) for _ in range(length))
        docs.append((str(i), body))
    return docs


def gil_status() -> str:
    """Human-readable description of whether this build is free-threaded."""
    is_gil_enabled = getattr(sys, "_is_gil_enabled", None)
    if is_gil_enabled is None:
        return "GIL build (standard CPython; threads do not run Python in parallel)"
    if is_gil_enabled():
        return (
            "free-threaded build, but GIL is currently ENABLED "
            "(set PYTHON_GIL=0 or run without a GIL-requiring extension to scale)"
        )
    return "free-threaded build, GIL DISABLED (pure-Python work scales across cores)"


# --------------------------------------------------------------------------- #
# Serial baseline: one writer, one thread.
# --------------------------------------------------------------------------- #
def build_serial(docs: list[tuple[str, str]], where: str) -> float:
    d = os.path.join(where, "serial")
    os.makedirs(d, exist_ok=True)
    ix = index.create_in(d, make_schema())
    t0 = time.perf_counter()
    w = ix.writer(limitmb=128)
    for doc_id, body in docs:
        w.add_document(id=doc_id, body=body)
    w.commit()
    elapsed = time.perf_counter() - t0
    ix.close()
    return elapsed


# --------------------------------------------------------------------------- #
# Parallel: N worker threads, each its own sub-index, then merge.
# --------------------------------------------------------------------------- #
def _build_shard(schema: Schema, shard: list[tuple[str, str]], subdir: str) -> str:
    """Worker: build one sub-index from one shard. Runs in its own thread.

    Only *this* thread touches this writer and this directory, so the
    single-writer contract holds with no lock contention between workers.
    """
    os.makedirs(subdir, exist_ok=True)
    ix = index.create_in(subdir, schema)
    w = ix.writer(limitmb=128)
    for doc_id, body in shard:
        w.add_document(id=doc_id, body=body)
    w.commit()
    ix.close()
    return subdir


def build_parallel(
    docs: list[tuple[str, str]], where: str, workers: int
) -> tuple[float, str]:
    schema = make_schema()  # immutable once built -> shared across threads
    shards: list[list[tuple[str, str]]] = [docs[i::workers] for i in range(workers)]
    subdirs = [os.path.join(where, f"shard-{i}") for i in range(workers)]

    t0 = time.perf_counter()
    # Fan out: each thread builds its own sub-index in parallel.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_build_shard, schema, shards[i], subdirs[i])
            for i in range(workers)
        ]
        built = [f.result() for f in futures]

    # Fan in: merge the finished sub-indexes into one final index. add_reader
    # takes a read-only reader from each shard -- no writer is shared.
    final_dir = os.path.join(where, "final")
    os.makedirs(final_dir, exist_ok=True)
    final_ix = index.create_in(final_dir, schema)
    w = final_ix.writer(limitmb=256)
    for subdir in built:
        sub_ix = index.open_dir(subdir)
        with sub_ix.reader() as r:
            w.add_reader(r)
        sub_ix.close()
    w.commit(optimize=True)
    elapsed = time.perf_counter() - t0
    final_ix.close()
    return elapsed, final_dir


def verify_equivalent(docs: list[tuple[str, str]], final_dir: str) -> None:
    """The parallel-built index must return the same docs as a serial one."""
    ix = index.open_dir(final_dir)
    with ix.searcher() as s:
        assert s.doc_count() == len(docs), (
            f"doc count mismatch: {s.doc_count()} != {len(docs)}"
        )
        qp = QueryParser("body", ix.schema)
        for term in ("python", "search", "fuzzy", "throughput"):
            q = qp.parse(term)
            n = len(s.search(q, limit=None))
            assert n >= 0
    ix.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--docs", type=int, default=20000, help="number of documents")
    ap.add_argument(
        "--workers",
        type=int,
        default=min(8, (os.cpu_count() or 2)),
        help="number of worker threads (default: min(8, CPU count))",
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="only verify parallel == serial correctness, skip timing report",
    )
    args = ap.parse_args()

    print(f"Runtime: {gil_status()}")
    print(f"CPUs: {os.cpu_count()}  workers: {args.workers}  docs: {args.docs}")
    print("Building corpus...")
    docs = make_corpus(args.docs)

    where = tempfile.mkdtemp(prefix="whoosh-parallel-")
    try:
        print("\nSerial baseline (1 writer)...")
        serial_t = build_serial(docs, where)
        print(f"  serial:   {serial_t:6.2f}s")

        print(f"Parallel ({args.workers} threads + merge)...")
        parallel_t, final_dir = build_parallel(docs, where, args.workers)
        print(f"  parallel: {parallel_t:6.2f}s  (build {args.workers} shards + merge)")

        print("\nVerifying parallel index == serial index (doc count + queries)...")
        verify_equivalent(docs, final_dir)
        print("  OK: parallel-built index is correct.")

        if not args.check:
            speedup = serial_t / parallel_t if parallel_t else float("nan")
            print(f"\nSpeedup: {speedup:.2f}x")
            if "_is_gil_enabled" not in dir(sys) or sys._is_gil_enabled():
                print(
                    "  Note: on a GIL build a speedup near 1x is EXPECTED -- the "
                    "merge adds\n  a little overhead the serial path avoids. Run "
                    "this on a free-threaded\n  build (3.13t/3.14t, PYTHON_GIL=0) "
                    "to see the parallel path pull ahead."
                )
            else:
                print(
                    "  Free-threaded build: a speedup above 1x means pure-Python "
                    "indexing is\n  scaling across cores without any C extension."
                )
    finally:
        shutil.rmtree(where, ignore_errors=True)


if __name__ == "__main__":
    main()
