# Whoosh MCP server — serve a pure-Python full-text index to AI agents over MCP.
#
# Build:  docker build -t whoosh-mcp .
# Run:    docker run --rm -i -v "$HOME/notes:/corpus:ro" whoosh-mcp /corpus
#
# The server speaks the Model Context Protocol over stdio (-i keeps stdin open),
# so it is ready to be spawned by any MCP client (Claude Desktop, IDE agents, or
# a custom agent loop). With no argument it serves a few built-in sample
# documents so the image is introspectable out of the box.
FROM python:3.12-slim

# Metadata so registries can attribute the image.
LABEL org.opencontainers.image.title="whoosh-mcp" \
      org.opencontainers.image.description="Pure-Python full-text (BM25F) search over the Model Context Protocol." \
      org.opencontainers.image.source="https://github.com/priya-sundaram-dev/whoosh" \
      org.opencontainers.image.licenses="BSD-2-Clause"

WORKDIR /app

# Install from the checked-out source so the image always matches this commit.
COPY . /app
RUN pip install --no-cache-dir ".[mcp]"

# stdio transport: an MCP client spawns the container and talks JSON-RPC over
# stdin/stdout. Pass a corpus directory as an argument, or mount one and set
# WHOOSH_MCP_CORPUS. With neither, the built-in sample documents are served.
ENTRYPOINT ["whoosh-mcp"]
