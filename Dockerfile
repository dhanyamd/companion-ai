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

# Copy your application code into the container
COPY src/ /app/

# Set the virtual environment environment variables
ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

# Install the package in editable mode
RUN uv pip install -e .

# Define volumes
VOLUME ["/app/data"]

# Expose the port
EXPOSE 8080

# Create a startup script with debugging information
RUN echo '#!/bin/bash\n\
echo "Starting FastAPI application..."\n\
echo "Current directory: $(pwd)"\n\
echo "Python path: $PYTHONPATH"\n\
echo "Available modules:"\n\
python -c "import sys; print(\"\\n\".join(sys.path))"\n\
echo "Testing module import:"\n\
python -c "from ai_companion.interfaces.whatsapp.webhook_endpoint import app; print(\"App imported successfully\")"\n\
exec uvicorn ai_companion.interfaces.whatsapp.webhook_endpoint:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1 --log-level debug\n\
' > /app/start.sh && chmod +x /app/start.sh

# Run the startup script
CMD ["/app/start.sh"]