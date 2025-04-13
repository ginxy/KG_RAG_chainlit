import os
from fastapi import FastAPI, Security
from fastapi.security import APIKeyHeader
from mcp import Tool, Resource, Prompt, MCPRouter
import logging
from src.db_operations import DBOps
from src.upload_data import DataUploader
from contextlib import asynccontextmanager
from fastapi import FastAPI
from mcp import MCPHost
from src.llm_processor import LLMProcessor
from typing import List, Annotated

logger = logging.getLogger(__name__)

mcp_router = MCPRouter()
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern lifespan handler replacing deprecated on_event"""
    # Startup
    await register_assets()
    yield
    # Shutdown logic here

app = FastAPI(
    title="MCP Server",
    lifespan=lifespan,
    openapi_url="/mcp/openapi.json",
    docs_url="/mcp/docs"
)

# ---------- Tools ----------
@mcp_router.tool(
    name="augmented_retrieval",  # Corrected name to match functionality
    description="Hybrid knowledge graph retrieval with vector search",
    params={
        "query": {"type": "string"},
        "score_threshold": {"type": "float", "default": 0.15}
    }
)
async def augmented_retrieval(query: str, score_threshold: float):
    """Properly named implementation"""
    kg = DBOps()
    await kg.connect()
    try:
        return await kg.augmented_retrieval(query, score_threshold)
    finally:
        await kg.close()


@mcp_router.tool(name="pdf_ingestion", description="Process PDF documents into knowledge graph and vector store",
    params={
        "file_path": {"type": "string"}, "original_filename": {"type": "string"}
        })
async def ingest_pdf(file_path: str, original_filename: str):
    uploader = DataUploader()
    await uploader.initialize()
    try:
        return await uploader.upload_data(file_path)
    finally:
        await uploader.close()


# ---------- Resources ----------
@mcp_router.resource(name="model_registry", description="Access to configured LLM models and their parameters")
def get_model_registry():
    from src.model_registry import ModelRegistry
    return ModelRegistry().config


# ---------- Prompts ----------
@mcp_router.prompt(name="kg_qa_template", template="""[CONTEXT]
    {context}

    [QUESTION]
    {query}

    [INSTRUCTIONS]
    - Use exact terms from context
    - Cite sources when available""")
def kg_qa_prompt(context: str, query: str) -> str:
    pass


app.include_router(mcp_router, prefix="/mcp", dependencies=[Security(api_key_header)])

async def register_assets():
    """Register all MCP components with the host"""
    from mcp import MCPHost
    host = MCPHost(os.getenv("MCP_HOST", "http://host:7432"))

    await host.register(tools=[augmented_retrieval, ingest_pdf], resources=[get_model_registry], prompts=[kg_qa_prompt],
        auth_token=os.getenv("MCP_AUTH_TOKEN"))
    logger.info("MCP assets registered successfully")