# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (Fixed for Debian Trixie)
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Upgrade pip first to the latest version for better connection handling
RUN pip install --upgrade pip

# Copy the requirements file into the container
COPY requirements.txt .

# [NUCLEAR OPTION] Increase timeout and add retries for unstable internet
RUN pip install --default-timeout=1000 --retries 10 --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Run the application using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7806"]