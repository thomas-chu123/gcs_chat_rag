# Use the official Python base image
FROM python:3.9-slim
# Set the working directory in the container
WORKDIR /app
# Copy the dependencies file to the working directory
COPY requirements.txt .
# Install any dependencies
RUN pip install --no-cache-dir -r requirements.txt
# Copy the content of the local src directory to the working directory
COPY app.py .
# Specify the command to run on container start
CMD ["streamlit", "run", "app.py", "--server.port=8501","--server.address=0.0.0.0"]