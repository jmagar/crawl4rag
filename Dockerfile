FROM python:3.12-slim

ARG PORT=8051
ARG USER_ID=99
ARG GROUP_ID=100

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

# Create non-root user and group
RUN (getent group ${GROUP_ID} || groupadd -g ${GROUP_ID} appgroup) && \
    useradd -u ${USER_ID} -g ${GROUP_ID} -d /app -s /sbin/nologin -c "Docker image user" appuser

# Set up Playwright browser path
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

# Install uv as root temporarily
RUN pip install uv

# Copy only the dependency and metadata files first to leverage Docker cache
COPY pyproject.toml README.md ./

# Install Python packages. This is a cached layer.
RUN uv pip install --system -e .

# Copy the rest of the application files.
COPY . .

# Set up the application directory and all necessary cache directories with correct permissions
# This prevents all filesystem permission errors for subsequent steps.
RUN mkdir -p /app/.crawl4ai ${PLAYWRIGHT_BROWSERS_PATH} && \
    chown -R appuser:${GROUP_ID} /app ${PLAYWRIGHT_BROWSERS_PATH}

# Switch to the non-root user
USER appuser

# Now, install Playwright browsers as the correct user.
# The cache directory is already created with the correct permissions.
RUN playwright install chromium

# Create health check endpoint
EXPOSE ${PORT}

# Command to run the MCP server
CMD ["python", "src/crawl4ai_mcp.py"]
