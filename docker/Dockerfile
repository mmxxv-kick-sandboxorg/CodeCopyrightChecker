# Dockerfile
FROM ubuntu:22.04

# required package install
RUN apt-get update && apt-get install -y \
    python3 python3-pip git curl gh && \
    rm -rf /var/lib/apt/lists/*

# Hugging Face Transformers and dependency package install
RUN pip3 install --no-cache-dir transformers torch accelerate

# work directory
WORKDIR /app

# copy check script to container
COPY scripts/gitDiffCheck.py /app/gitDiffCheck.py

CMD ["python3", "gitDiffCheck.py"]

