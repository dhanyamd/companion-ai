# Use an appropriate base image
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Install the project into `/app`
WORKDIR /app

# Set environment variables (e.g., set Python to run in unbuffered mode)
ENV PYTHONUNBUFFERED 1

# Install system dependencies for building libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy the dependency management files (lock file and pyproject.toml) first
COPY uv.lock pyproject.toml README.md /app/

# Install the application dependencies
RUN uv sync --frozen --no-cache

# Copy your application code and necessary directories into the container
COPY src/ /app/src/
COPY models/ /app/models/
COPY short_term_memory/ /app/short_term_memory/
COPY long_term_memory/ /app/long_term_memory/

# Set the virtual environment environment variables
ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

# Install the package in editable mode
RUN uv pip install -e .

# Define volumes
VOLUME ["/app/data"]

# Expose the port
EXPOSE 8080

# Command to run the application
# Use exec form to allow graceful shutdown signals
CMD ["/app/.venv/bin/uvicorn", "ai_companion.interfaces.whatsapp.webhook_endpoint:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--log-level", "debug"]