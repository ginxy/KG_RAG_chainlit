from fastapi import FastAPI, HTTPException, status, Security, BackgroundTasks
from fastapi.security import APIKeyHeader
from neo4j import AsyncGraphDatabase
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import os
import logging
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

# Initialize FastAPI app with proper lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Validate environment variables on startup
    required_env_vars = ["NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD", "MCP_AUTH_TOKEN"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing_vars)}")
    yield

app = FastAPI(lifespan=lifespan)

# Security setup
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_token(api_key: str = Security(api_key_header)):
    if api_key != os.getenv("MCP_AUTH_TOKEN"):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API Key")
    return True


# Tool implementation - replaced mcp.Tool
@app.post("/tools/vector_search")
async def vector_search(query: str, top_k: int = 5, _=Security(verify_token)):
    qdrant = QdrantClient(host=os.getenv("QDRANT_HOST"), port=6333)
    encoder = SentenceTransformer('all-MiniLM-L6-v2')

    try:
        embedding = encoder.encode(query)
        results = qdrant.query_points(collection_name="doc_chunks", query_vector=embedding.tolist(), limit=top_k)
        return [{
            "text": hit.payload["text"], "score": float(hit.score), "source": hit.payload.get("source", "unknown")
            } for hit in results]
    except Exception as e:
        logger.error(f"Vector search failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Search failed")


class ProgressContext:
    """Replacement for MCP ProgressContext"""

    def __init__(self, task_id: str = None):
        self.task_id = task_id
        self.progress = 0
        self.total_steps = 0

    async def report_progress(self, current: int, total: int, message: str):
        self.progress = current / total
        logger.info(f"[Task {self.task_id}] {message} (Progress: {current}/{total})")


@app.post("/tools/graph_query")
async def graph_query(cypher: str, params: dict,  background_tasks: BackgroundTasks, _=Security(verify_token)):
    """Updated implementation with progress tracking"""
    ctx = ProgressContext()

    try:
        # Validate query
        cypher = cypher.strip().lower()
        if not cypher.startswith(("match", "call", "return")) or "create" in cypher:
            raise ValueError("Only read operations allowed")

        # Async execution with progress reporting
        async def execute_query():
            try:
                await ctx.report_progress(1, 4, "Connecting to Neo4j")
                driver = AsyncGraphDatabase.driver(os.getenv("NEO4J_URI"),
                    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")), max_connection_pool_size=5)

                await ctx.report_progress(2, 4, "Executing query")
                async with driver.session() as session:
                    result = await session.run(cypher, parameters=params)
                    data = await result.data()
                    summary = await result.consume()

                    await ctx.report_progress(3, 4, "Processing results")
                    return {
                        "nodes": [dict(record) for record in data], "summary": {
                            "nodes_created"        : summary.counters.nodes_created,
                            "relationships_created": summary.counters.relationships_created
                            }
                        }
            finally:
                await ctx.report_progress(4, 4, "Closing connection")
                await driver.close()

        # Run query execution in background
        background_tasks.add_task(execute_query)
        return {"status": "started", "message": "Query processing in progress"}

    except Exception as e:
        logger.error(f"Graph query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tools/model_config")
def get_model_config():
    """Proper resource implementation"""
    try:
        from src.model_registry import ModelRegistry
        return ModelRegistry().config
    except Exception as e:
        logger.error(f"Failed to load model config: {str(e)}")
        return {"error": "Model configuration unavailable"}


# Health check endpoint with proper dependencies
@app.get("/mcp/health", dependencies=[Security(verify_token)])
async def health_check():
    return {
        "status": "ok", "services": {
            "neo4j": await check_neo4j_connection(), "qdrant": check_qdrant_connection()
            }
        }


async def check_neo4j_connection():
    driver = None
    try:
        driver = AsyncGraphDatabase.driver(os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")), connection_timeout=5)
        async with driver.session() as session:
            await session.run("RETURN 1")
        return {"status": "connected", "version": await get_neo4j_version(driver)}
    except Exception as e:
        return {"status": "disconnected", "error": str(e)}
    finally:
        if driver:
            await driver.close()


async def get_neo4j_version(driver):
    try:
        async with driver.session() as session:
            result = await session.run("CALL dbms.components() YIELD name, versions RETURN name, versions")
            data = await result.data()
            return data[0]["versions"][0] if data else "unknown"
    except Exception:
        return "unknown"


def check_qdrant_connection():
    try:
        client = QdrantClient(host="qdrant", port=6333, timeout=5)
        collections = client.get_collections()
        return {
            "status"     : "connected",
            "collections": len(collections.collections),
            "version"    : client.http.client.headers.get("x-qdrant-version", "unknown")
            }
    except Exception as e:
        return {"status": "disconnected", "error": str(e)}


# Register MCP router with security dependencies
# app.include_router(mcp_router, prefix="/mcp", dependencies=[Security(verify_token)])