# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (including curl for healthcheck and git-lfs)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libcairo2-dev \
    libpango1.0-dev \
    libjpeg-dev \
    libgif-dev \
    librsvg2-dev \
    curl \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers (for PDF generation)
RUN playwright install chromium --with-deps || playwright install chromium

# Copy application files (including .git for LFS)
COPY . .

# Initialize Git LFS and pull actual LFS files
# Railway clones with LFS pointers, we need to pull the actual files
RUN git lfs install && \
    if [ -d .git ]; then \
        git lfs pull; \
    else \
        echo "Warning: .git directory not found - LFS files may not be available"; \
    fi

# Make startup script executable and ensure Unix line endings
RUN sed -i 's/\r$//' start.sh 2>/dev/null || true && chmod +x start.sh

# Expose Streamlit port (Railway will set PORT env var)
EXPOSE 8501

# Health check (Railway will handle port mapping)
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit using startup script (handles PORT env var from Railway)
CMD ["/bin/bash", "start.sh"]


