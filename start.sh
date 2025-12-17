#!/bin/bash
# Startup script for Railway deployment
# Railway sets PORT environment variable

PORT=${PORT:-8501}

exec streamlit run streamlit_app.py \
    --server.port=$PORT \
    --server.address=0.0.0.0 \
    --server.headless=true
