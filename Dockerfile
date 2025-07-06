FROM python:3.12-slim

ARG PORT=8051
ARG USER_ID=1000
ARG GROUP_ID=1000

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        curl \
        ca-certificates \
        libglib2.0-0 \
        libnss3 \
        libatk-bridge2.0-0 \
        libdrm2 \
        libxkbcommon0 \
        libgtk-3-0 \
        libgbm1 \
        libasound2 \
        libatspi2.0-0 \
        libx11-6 \
        libxcomposite1 \
        libxdamage1 \
        libxext6 \
        libxfixes3 \
        libxrandr2 \
        libcairo2 \
        libpango-1.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user with force flags for rebuild safety
RUN groupadd -f -g ${GROUP_ID} appuser && \
    useradd -r -u ${USER_ID} -g appuser -d /app -s /sbin/nologin \
    -c "Docker image user" appuser || true

# Install uv as root first
RUN pip install uv

# Copy the MCP server files and set ownership
COPY --chown=appuser:appuser . .

# Create directories that crawl4ai needs and set permissions
RUN mkdir -p /app/.crawl4ai && \
    chown -R appuser:appuser /app/.crawl4ai

# Create cache directory for Playwright
RUN mkdir -p /app/.cache && \
    chown -R appuser:appuser /app/.cache

# Install packages directly to the system (no virtual environment)
# Combining commands to reduce Docker layers
RUN uv pip install --system -e .

# Change to non-root user
USER appuser

# Install Playwright browsers as appuser
RUN playwright install chromium

# Create health check endpoint
EXPOSE ${PORT}

# Command to run the MCP server
CMD ["python", "src/crawl4ai_mcp.py"]
