# Multi-stage build for AI Developer Agent
# Stage 1: Node.js base — install MCP servers here
FROM node:20-bookworm-slim AS node-base

# Install GitHub MCP server globally (GitLab/Bitbucket use direct REST API clients)
RUN npm install -g @modelcontextprotocol/server-github

# Stage 2: Final image
FROM python:3.12-bookworm

# Copy Node.js runtime from node image
COPY --from=node-base /usr/local/bin/node /usr/local/bin/node
COPY --from=node-base /usr/local/lib/node_modules /usr/local/lib/node_modules

# Create npm/npx shims that point to the correct node_modules
RUN ln -sf /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm && \
    ln -sf /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx && \
    chmod +x /usr/local/bin/npm /usr/local/bin/npx

WORKDIR /app

# Copy dependency files first for layer caching
COPY pyproject.toml README.md ./
COPY mcp_agent.config.yaml ./

# Install Python dependencies (non-editable for Docker)
RUN pip install --no-cache-dir ".[dev]"

# Copy source code
COPY src/ src/
COPY tests/ tests/
COPY scripts/ scripts/
COPY .env.example .env.example

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import httpx; print(httpx.get('http://localhost:8000/health').status_code)" || exit 1

CMD ["uvicorn", "src.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
