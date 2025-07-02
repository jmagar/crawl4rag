FROM python:3.12-slim

ARG PORT=8051
ARG USER_ID=1000
ARG GROUP_ID=1000

WORKDIR /app

# Install system dependencies with optimization
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user with force flags for rebuild safety
RUN groupadd -f -g ${GROUP_ID} appuser && \
    useradd -r -u ${USER_ID} -g appuser -d /app -s /sbin/nologin \
    -c "Docker image user" appuser || true

# Install uv as root first
RUN pip install uv

# Copy the MCP server files and set ownership
COPY --chown=appuser:appuser . .

# Install packages directly to the system (no virtual environment)
# Combining commands to reduce Docker layers
RUN uv pip install --system -e . && \
    python -m crawl4ai.async_crawler --setup

# Change to non-root user
USER appuser

# Create health check endpoint
EXPOSE ${PORT}

# Add health check using curl instead of Python
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:${PORT}/health || exit 1

# Command to run the MCP server
CMD ["python", "src/crawl4ai_mcp.py"]
