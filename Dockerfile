FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies if required (e.g., for building some C extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose the default Hugging Face Spaces port
EXPOSE 7860

# Create a non-root user (Hugging Face Spaces requirement/best practice)
RUN useradd -m -u 1000 user
USER user

# Define environment variables
ENV PYTHONUNBUFFERED=1

# Command to run the application (run.py already handles uvicorn binding to 0.0.0.0:7860)
CMD ["python", "run.py"]
