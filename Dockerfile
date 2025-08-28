# NeuroGrade Pro - Brain Tumor Analysis Platform
# Optimized Docker container for medical AI application

FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables for Python and Streamlit
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Debug: List files to verify they're copied
RUN ls -la /app/ && echo "=== Checking critical files ===" && ls -la /app/*.h5 /app/*.zip || echo "Critical files missing!"

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash neurograde && \
    chown -R neurograde:neurograde /app
USER neurograde

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]