"""Expose a Whoosh full-text index to LLM agents as an MCP server.

The Model Context Protocol (MCP) lets AI agents (Claude Desktop, IDE assistants,
OpenAI/Anthropic connectors, custom agent loops) call tools over a standard
protocol. Because Whoosh is pure Python and embeds a real BM25F index as a plain
directory of files, it makes an excellent *local, no-server, no-native-deps*
search backend for an agent's "search" and "fetch" tools.

This example builds a tiny index and serves two tools that follow the common
connector convention::

    search(query, limit) -> list of {id, title, score, snippet}
    fetch(id)            -> full document text for a given id

Run it as a real MCP server (requires the official SDK: ``pip install mcp``)::

    pip install whoosh3 mcp
    python examples/mcp_server.py            # stdio transport, ready for an agent

Point an MCP client at it (e.g. in Claude Desktop's config)::

    {
      "mcpServers": {
        "whoosh": { "command": "python", "args": ["/path/to/examples/mcp_server.py"] }
      }
    }

The search core (SearchCore) has no MCP dependency, so you can import and
unit-test it directly, or reuse it behind any agent framework (LangChain /
LlamaIndex tools, OpenAI function calling, or a plain function).
"""

from __future__ import annotations

import os.path
import tempfile
from dataclasses import dataclass

from whoosh import highlight
from whoosh.fields import ID, TEXT, Schema
from whoosh.index import create_in, open_dir
from whoosh.qparser import MultifieldParser

# A few sample documents. Swap in your own corpus / directory of files.
DOCS = [
    {
        "id": "py-gil",
        "title": "The Python GIL",
        "body": "The global interpreter lock serializes bytecode execution so only "
                "one thread runs Python at a time. CPython 3.13 ships an experimental "
                "free-threaded build that can disable the GIL.",
    },
    {
        "id": "bm25",
        "title": "BM25 ranking",
        "body": "BM25 is a probabilistic ranking function that scores documents by "
                "term frequency and inverse document frequency with length "
                "normalization. Whoosh uses BM25F by default.",
    },
    {
        "id": "mcp",
        "title": "Model Context Protocol",
        "body": "MCP is an open protocol that standardizes how applications provide "
                "context and tools to large language models. Servers expose tools "
                "like search and fetch that an agent can call.",
    },
    {
        "id": "embedded",
        "title": "Embedded search engines",
        "body": "An embedded search engine runs inside your process instead of a "
                "separate server. Whoosh stores its index as a directory of files, "
                "so it deploys anywhere CPython runs, including serverless and CI.",
    },
]


@dataclass
class SearchCore:
    """A reusable Whoosh-backed search core with no MCP/agent dependency."""

    index_dir: str

    @classmethod
    def build(cls, docs=DOCS, index_dir=None):
        index_dir = index_dir or tempfile.mkdtemp(prefix="whoosh_mcp_")
        if not os.path.exists(index_dir):
            os.makedirs(index_dir)
        schema = Schema(
            id=ID(stored=True, unique=True),
            title=TEXT(stored=True),
            body=TEXT(stored=True),
        )
        ix = create_in(index_dir, schema)
        writer = ix.writer()
        for d in docs:
            writer.update_document(id=d["id"], title=d["title"], body=d["body"])
        writer.commit()
        return cls(index_dir=index_dir)

    def search(self, query, limit=5):
        ix = open_dir(self.index_dir)
        parser = MultifieldParser(["title", "body"], schema=ix.schema)
        q = parser.parse(query)
        out = []
        with ix.searcher() as searcher:
            results = searcher.search(q, limit=limit)
            results.fragmenter = highlight.ContextFragmenter(maxchars=160, surround=40)
            results.formatter = highlight.UppercaseFormatter()
            out.extend(
                {
                    "id": hit["id"],
                    "title": hit["title"],
                    "score": round(hit.score, 4),
                    "snippet": hit.highlights("body") or hit["body"][:160],
                }
                for hit in results
            )
        return out

    def fetch(self, doc_id):
        ix = open_dir(self.index_dir)
        with ix.searcher() as searcher:
            hit = searcher.document(id=doc_id)
            if hit is None:
                return {"id": doc_id, "error": "not found"}
            return {"id": hit["id"], "title": hit["title"], "text": hit["body"]}


def build_mcp_server(core=None):
    """Wrap a SearchCore in a FastMCP server exposing search + fetch tools."""
    from mcp.server.fastmcp import FastMCP  # noqa: PLC0415  (pip install mcp)

    core = core or SearchCore.build()
    mcp = FastMCP("whoosh-search")

    @mcp.tool()
    def search(query: str, limit: int = 5) -> list[dict]:
        """Full-text search the local corpus. Returns ranked {id, title, score, snippet}."""
        return core.search(query, limit)

    @mcp.tool()
    def fetch(id: str) -> dict:
        """Fetch the full text of a document by its id (as returned by search)."""
        return core.fetch(id)

    return mcp


if __name__ == "__main__":
    # Runs an MCP server over stdio, ready to be spawned by an MCP client.
    build_mcp_server().run()
