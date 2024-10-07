FROM python:3.9


WORKDIR /script

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY batch_chunk_and_load.py .

# # Make port 8080 available to the world outside this container
# EXPOSE 8080

# # Define environment variable
# ENV NAME World

ENTRYPOINT [ "python3", "batch_chunk_and_load.py"]
