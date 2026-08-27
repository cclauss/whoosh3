"""Smoke tests for examples/parallel_indexing.py.

The parallel-indexing recipe is the *blessed pattern* the concurrency guide
points free-threaded (no-GIL) users at: fan out into one sub-index per worker
thread, then fan in with ``writer.add_reader()``. The whole promise is that the
merged result is a correct, ordinary Whoosh index -- identical to what a single
serial writer would produce. These tests pin that correctness so a refactor of
the writer/merge path can't silently break the pattern (timing is not
asserted; a speedup only shows up on an actual free-threaded build).
"""

import importlib.util
import pathlib
import tempfile

import pytest

_EXAMPLE = (
    pathlib.Path(__file__).resolve().parent.parent / "examples" / "parallel_indexing.py"
)


@pytest.fixture(scope="module")
def ex():
    spec = importlib.util.spec_from_file_location("parallel_indexing", _EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_parallel_index_matches_serial_doc_count(ex):
    docs = ex.make_corpus(400)
    with tempfile.TemporaryDirectory() as where:
        _, final_dir = ex.build_parallel(docs, where, workers=4)
        # verify_equivalent asserts doc count == len(docs) and queries run.
        ex.verify_equivalent(docs, final_dir)


def test_uneven_shards_lose_no_documents(ex):
    # 401 docs across 4 threads => shards of 101/100/100/100. Every document
    # must survive the split + merge, unique id field and all.
    docs = ex.make_corpus(401)
    with tempfile.TemporaryDirectory() as where:
        _, final_dir = ex.build_parallel(docs, where, workers=4)
        ex.verify_equivalent(docs, final_dir)


def test_single_worker_is_still_correct(ex):
    # Degenerate fan-out (1 worker) must behave like a plain build + merge.
    docs = ex.make_corpus(150)
    with tempfile.TemporaryDirectory() as where:
        _, final_dir = ex.build_parallel(docs, where, workers=1)
        ex.verify_equivalent(docs, final_dir)


def test_gil_status_is_descriptive(ex):
    # The script must always tell the user which runtime they are on -- the
    # whole point is that the speedup only appears without the GIL.
    status = ex.gil_status()
    assert isinstance(status, str) and status
