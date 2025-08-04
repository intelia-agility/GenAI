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

# Specify the command to run your application using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

# Expose the port the app runs on
EXPOSE 8080

