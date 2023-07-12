FROM ubuntu:latest

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY model.py requirements.txt /app/

RUN pip3 install --no-cache-dir -r requirements.txt

CMD ["python3", "model.py"]