# Development Dockerfile
# Assignment 3: HuggingFace Model Training & Docker Deployment

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Verify installations
RUN python -c "import transformers; import torch; import sklearn; print('All dependencies OK')"

# Default command: show help
CMD ["python", "main.py", "--help"]
