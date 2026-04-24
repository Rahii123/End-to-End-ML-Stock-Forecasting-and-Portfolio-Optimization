# Use lightweight Python 3.11 slim base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt ./

# Install CPU-Only PyTorch first (pre-compiled wheel, no compilation needed)
RUN pip install --no-cache-dir \
    torch==2.1.0+cpu \
    torchvision==0.16.0+cpu \
    torchaudio==2.1.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining application dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Start the Streamlit app
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
