import os
import sys
import concurrent.futures
from db_operations import DBOps
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import pandas as pd
import json
import asyncio
from typing import Union, Dict, List
from mcp import MCPClient
from pathlib import Path
from src_utils import setup_logger
logger = setup_logger("data_upload")


class DataUploader:
    def __init__(self):
        self.kg = DBOps()
        self.qdrant = QdrantClient(host="qdrant", port=6333)
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.mcp = MCPClient(host=os.getenv("MCP_HOST"), auth_token=os.getenv("MCP_AUTH_TOKEN"), retries=3)

    async def initialize(self):
        """Initialize connections to both databases"""
        if not await self.kg.connect():
            raise ConnectionError("Failed to connect to Neo4j")
        if not self.qdrant.get_collections():
            raise ConnectionError("Failed to connect to Qdrant")
        logger.info("Connected to both Neo4j and Qdrant")

    async def upload_data(self, file_path: Union[str, Path], data_type: str = "auto") -> str:
        """Maintain original return format"""
        try:
            result = await self.mcp.execute_tool("pdf_ingestion", params={
                "file_path": file_path, "original_filename": os.path.basename(file_path)
                }, timeout=120  # Longer timeout for files
                )

            # Preserve original success message format
            return (f"Successfully processed PDF: extracted {result['characters']} characters, "
                    f"identified {result['entities']} entities, created {result['chunks']} chunks.")

        except MCPTimeoutError:
            return "Error: PDF processing timed out"
        except Exception as e:
            return f"Error processing PDF: {str(e)}"
        # """
        # Unified data uploader that handles multiple formats:
        # - CSV: Structured knowledge graph data
        # - JSON: Semi-structured data with entities/relationships
        # - TXT/PDF: Raw text documents (processed into chunks)
        # """
        # file_path = Path(file_path)
        # stats = {"neo4j_nodes": 0, "qdrant_vectors": 0}
        #
        # if data_type == "auto":
        #     data_type = file_path.suffix[1:].lower()
        #
        # try:
        #     if data_type == "csv":
        #         stats.update(await self._process_csv(file_path))
        #     elif data_type == "json":
        #         stats.update(await self._process_json(file_path))
        #     elif data_type in ["txt", "pdf"]:
        #         stats.update(await self._process_document(file_path))
        #     else:
        #         raise ValueError(f"Unsupported file type: {data_type}")
        #
        #     logger.info(f"Upload completed: {stats}")
        #     # return stats
        #     return await self.mcp.execute_tool("pdf_ingestion", file_path=file_path,
        #         original_filename=os.path.basename(file_path))
        # except Exception as e:
        #     logger.error(f"Upload failed: {str(e)}")
        #     raise

    async def _process_csv(self, file_path: Path) -> Dict[str, int]:
        """Process CSV files with structured relationships"""
        df = pd.read_csv(file_path)

        # Upload to Neo4j
        await self.kg.ingest_data("csv", str(file_path))

        # Create text chunks for Qdrant
        chunks = self._create_csv_chunks(df)
        return await self._upload_chunks_to_qdrant(chunks, source=file_path.name)

    async def _process_json(self, file_path: Path) -> Dict[str, int]:
        """Process JSON files with entities and relationships"""
        with open(file_path) as f:
            data = json.load(f)

        # Upload to Neo4j
        await self.kg.ingest_data("json", str(file_path))

        # Create text chunks for Qdrant
        chunks = self._create_json_chunks(data)
        return await self._upload_chunks_to_qdrant(chunks, source=file_path.name)

    async def _process_document(self, file_path: Path) -> Dict[str, int]:
        """Process text/PDF documents using existing PDF pipeline"""
        # Uses the same ingestion as PDF uploads through the UI
        result = await self.kg.ingest_pdf(str(file_path), concurrent.futures.ThreadPoolExecutor(),
            original_filename=file_path.name)
        logger.info(f"Document processing result: {result}")
        return {"neo4j_nodes": "N/A", "qdrant_vectors": "N/A"}  # Stats handled internally

    def _create_csv_chunks(self, df: pd.DataFrame) -> List[str]:
        """Convert CSV relationships into natural language chunks"""
        chunks = []
        for _, row in df.iterrows():
            chunk = (f"{row['source']} {row['relationship'].lower()} "
                     f"{row['target']}. {row.get('description', '')}")
            chunks.append(chunk)
        return chunks

    def _create_json_chunks(self, data: Dict) -> List[str]:
        """Convert JSON entities into descriptive chunks"""
        chunks = []
        for entity in data.get('entities', []):
            chunk = f"{entity['name']}: {entity['description']}"
            chunks.append(chunk)
        return chunks

    async def _upload_chunks_to_qdrant(self, chunks: List[str], source: str) -> Dict[str, int]:
        """Upload text chunks to Qdrant with embeddings"""
        if not chunks:
            return {"qdrant_vectors": 0}

        embeddings = self.encoder.encode(chunks)

        points = [{
            "id": idx, "vector": emb.tolist(), "payload": {
                "text": chunk, "source": source, "type": "structured_data"
                }
            } for idx, (chunk, emb) in enumerate(zip(chunks, embeddings))]

        self.qdrant.upsert(collection_name="doc_chunks", points=points)
        return {"qdrant_vectors": len(points)}

    async def close(self):
        """Clean up resources"""
        await self.kg.close()


async def main(path_to_doc : str = None):
    uploader = DataUploader()
    try:
        await uploader.initialize()
        if path_to_doc:
            await uploader.upload_data(path_to_doc)
        else:
            await uploader.upload_data("data/tokenization-and-decentralized-business-models.pdf")
    finally:
        await uploader.close()


if __name__ == "__main__":
    arg1_path = sys.argv[1]
    asyncio.run(main(arg1_path))