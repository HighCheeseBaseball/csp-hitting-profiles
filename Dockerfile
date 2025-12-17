# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (including curl for healthcheck)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libcairo2-dev \
    libpango1.0-dev \
    libjpeg-dev \
    libgif-dev \
    librsvg2-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install gdown for downloading data file during build
RUN pip install --no-cache-dir gdown

# Install Playwright browsers (for PDF generation)
RUN playwright install chromium --with-deps || playwright install chromium

# Copy application files
COPY . .

# Download data file during build to avoid runtime timeouts
# Set MLB_DATA_URL as build arg (Railway will pass the env var as build arg)
ARG MLB_DATA_URL
RUN if [ -n "$MLB_DATA_URL" ]; then \
        echo "Downloading data file during build..."; \
        file_id="$MLB_DATA_URL"; \
        # If it's a full URL, extract the ID
        if echo "$MLB_DATA_URL" | grep -q "drive.google.com"; then \
            file_id=$(echo "$MLB_DATA_URL" | grep -oE '/d/([a-zA-Z0-9_-]+)' | cut -d'/' -f3); \
        fi; \
        if [ -n "$file_id" ]; then \
            echo "Downloading file ID: $file_id"; \
            gdown "https://drive.google.com/uc?id=$file_id" -O MLB_data.csv && \
            echo "Data file downloaded successfully during build" || \
            echo "Build-time download failed, will download at runtime"; \
        else \
            echo "Could not extract file ID from MLB_DATA_URL"; \
        fi; \
    else \
        echo "MLB_DATA_URL not set, will download at runtime if needed"; \
    fi

# Make startup script executable and ensure Unix line endings
RUN sed -i 's/\r$//' start.sh 2>/dev/null || true && chmod +x start.sh

# Expose Streamlit port (Railway will set PORT env var)
EXPOSE 8501

# Health check (Railway will handle port mapping)
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit using startup script (handles PORT env var from Railway)
CMD ["/bin/bash", "start.sh"]


