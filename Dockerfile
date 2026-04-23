# Use a lightweight Python base image
FROM python:3.12-slim

# Install system dependencies
# build-essential is required for compiling numeric libraries like numpy/scipy if needed
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Copy requirements file first for Docker layer caching
COPY requirements.txt ./

# CRITICAL SENIOR FIX: 
# Install CPU-Only PyTorch from the specific index URL first to prevent a massive 3GB GPU download.
# We use --no-cache-dir to keep the Docker image as small and cheap as possible.
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --no-cache-dir \
    && pip install -r requirements.txt --no-cache-dir

# Copy the rest of the application files
COPY . .

# Expose the port Streamlit uses
EXPOSE 8501

# Command to keep the Streamlit app running indefinitely on the AWS server
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
