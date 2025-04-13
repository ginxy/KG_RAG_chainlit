FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc python3-dev

# Install required packages (removed mcp)
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    neo4j>=5.12 \
    qdrant-client>=1.6.5 \
    sentence-transformers>=2.2.2 \
    httpx \
    python-dotenv

# Copy only mcp_server files
COPY mcp_server/ /app/
COPY config/models.yaml /app/config/

ENV PYTHONPATH="/app:${PYTHONPATH}"
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7432"]