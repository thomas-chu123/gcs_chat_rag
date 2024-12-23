#Dockerfile
FROM python:3.10-slim
ENV PYTHONUNBUFFERED True
ENV APP_HOME /app
WORKDIR $APP_HOME
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*
RUN apt-get update \
    && pip install --upgrade pip \
    && apt-get -y install libpq-dev gcc \
    && pip install psycopg2 \
    && pip install gunicorn \
    && pip install langchain \
    && pip install flask \
    && pip install google-cloud-aiplatform>=1.38.0 \
    && pip install psycopg2-binary \
    && pip install pgvector \
    && pip install numpy \
    && pip install pandas \
    && pip install unstructured==0.7.12
COPY . ./
RUN pip install --no-cache-dir -r requirements.txt
RUN python3 -m nltk.downloader punkt \
    && python3 -m nltk.downloader punkt_tab \
    && python3 -m nltk.downloader averaged_perceptron_tagger \
    && python3 -m nltk.downloader averaged_perceptron_tagger_eng
EXPOSE 8080
CMD exec gunicorn --bind 0.0.0.0:8080 --workers 1 --threads 8 --timeout 0 embed:app