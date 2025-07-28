
FROM ollama/ollama:0.10.0-rc2

RUN apt-get update && apt-get install -y \
    python3 python3-venv python3-pip build-essential poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

RUN pip install --no-cache-dir numpy spacy pymupdf ollama

RUN python -m spacy download en_core_web_sm

ENTRYPOINT []

CMD ollama serve & \
    sleep 5 && \
    ollama pull nomic-embed-text && \
    python3 process_collection.py