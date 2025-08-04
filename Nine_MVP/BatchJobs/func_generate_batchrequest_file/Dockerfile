FROM python:3.9-slim

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    libmediainfo0v5 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

#CMD ["python", "main.py"]


ENTRYPOINT [ "python3", "main.py"]
