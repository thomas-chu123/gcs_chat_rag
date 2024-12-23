# Use the official Python base image
FROM python:3.10-slim
# Set the working directory in the container
WORKDIR /app
# Copy the dependencies file to the working directory
COPY requirements.txt .
# Install any dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN python3 -m nltk.downloader punkt \
    && python3 -m nltk.downloader punkt_tab \
    && python3 -m nltk.downloader averaged_perceptron_tagger \
    && python3 -m nltk.downloader averaged_perceptron_tagger_eng
# Copy the content of the local src directory to the working directory
COPY app.py .
# Specify the command to run on container start
CMD ["streamlit", "run", "app.py", "--server.port=8501","--server.address=0.0.0.0"]