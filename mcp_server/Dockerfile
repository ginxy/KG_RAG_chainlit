FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY mcp_server/ /app/mcp_server/
COPY src/ /app/src/

ENV PYTHONPATH "${PYTHONPATH}:/app"
EXPOSE 7432

CMD ["uvicorn", "mcp_server.server:app", "--host", "0.0.0.0", "--port", "7432"]