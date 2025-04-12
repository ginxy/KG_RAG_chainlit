import asyncio
import concurrent.futures
import logging
import os
import re
import traceback
from datetime import datetime
from typing import List
from typing import Optional

import numpy as np
from neo4j import AsyncGraphDatabase
from neo4j.exceptions import AuthError
from neo4j.exceptions import ServiceUnavailable
from nltk.tokenize import sent_tokenize
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from tenacity import retry
from tenacity import stop_after_attempt
from tenacity import wait_fixed

from src_utils import async_error_handler
from src_utils import timing_decorator

logger = logging.getLogger(__name__)


class Neo4jKG:
    def __init__(self):
        self.driver = None
        self.ft_index_name = "entitySearch"
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.qdrant = QdrantClient(host="qdrant", port=6333)
        self._init_vector_db()

    def _init_vector_db(self):
        """Ensure Qdrant collection exists with correct schema"""
        from qdrant_client.http import models

        try:
            # Create collection if non-existent
            if not self.qdrant.collection_exists("doc_chunks"):
                self.qdrant.create_collection(collection_name="doc_chunks",
                    vectors_config=models.VectorParams(size=self.encoder.get_sentence_embedding_dimension(),
                        distance=models.Distance.COSINE))
                logger.info("Created Qdrant collection 'doc_chunks'")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant: {str(e)}")
            raise

    async def initialize(self):
        await self._create_constraints()
        await self._seed_initial_data()
        await self._create_ft_index()

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(2))
    async def connect(self):
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER")
        password = os.getenv("NEO4J_PASSWORD")

        logger.info(f"Connecting to Neo4j at {uri} with user {user}")

        try:
            self.driver = AsyncGraphDatabase.driver(uri, auth=(user, password), encrypted=False,
                                                    max_connection_pool_size=5, connection_timeout=30, keep_alive=True)

            async with self.driver.session() as session:
                await session.run("RETURN 1")

            logger.info("Neo4j connection established")
            return True

        except AuthError as e:
            logger.error(f"Authentication failed: {str(e)}")
            raise
        except ServiceUnavailable as e:
            logger.error(f"Service unavailable: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Connection error: {str(e)}")
            await self.driver.close()
            raise

    async def close(self):
        if self.driver:
            await self.driver.close()
            logger.info("Neo4j connection closed")

    async def _create_ft_index(self):
        index_query = """
            CREATE FULLTEXT INDEX entitySearch IF NOT EXISTS 
            FOR (e:Entity) ON EACH [e.name, e.description]
            OPTIONS { indexConfig: {
                `fulltext.analyzer`: 'english',
                `fulltext.eventually_consistent`: true
            }}
        """
        async with self.driver.session() as session:
            await session.run(index_query)

    async def _create_constraints(self):
        queries = ["CREATE CONSTRAINT IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
            "CREATE FULLTEXT INDEX entitySearch IF NOT EXISTS FOR (e:Entity) ON EACH [e.name, e.description]",
            "CREATE FULLTEXT INDEX chunkSearch IF NOT EXISTS FOR (c:Chunk) ON EACH [c.text]"]
        async with self.driver.session() as session:
            for query in queries:
                await session.run(query)

    async def _seed_initial_data(self):
        seed_data = [{"name": "AI", "description": "Artificial Intelligence"},
                     {"name": "ML", "description": "Machine Learning"}]
        async with self.driver.session() as session:
            await session.execute_write(self._create_entities, seed_data) # transaction object tx in callback function in 1st pos and injected by Neo4j

    async def augmented_retrieval(self, query: str, score_threshold: float = None) -> dict:
        """Hybrid search combining KG and vector results"""
        score_threshold = score_threshold or float(os.getenv("KG_SCORE_THRESHOLD", 0.15))
        max_results = int(os.getenv("MAX_RETRIEVAL_RESULTS", 10))
        diversity_penalty = float(os.getenv("SEARCH_DIVERSITY_PENALTY", 1.0))

        # Execute all searches in parallel
        kg_entities = self._search_entities(query, score_threshold)
        kg_chunks = self._search_chunks(query, score_threshold)
        vector_results = self.vector_search(query)

        # Wait for all results
        kg_entities, kg_chunks, vector_results = await asyncio.gather(kg_entities, kg_chunks, vector_results)

        # Convert vector results to match chunk format
        vector_chunks = [{
            "text": res["text"], "score": res["score"], "source": "vector"
            } for res in vector_results]

        # Combine and deduplicate chunks
        combined_chunks = self._combine_results(kg_chunks=kg_chunks, vector_chunks=vector_chunks,
                                                diversity_penalty=diversity_penalty, max_results=max_results)

        # Return in original format with additional source metadata
        return {
            "entities": kg_entities, "chunks": combined_chunks
            }

    def _combine_results(self, kg_chunks: List[dict], vector_chunks: List[dict], diversity_penalty: float,
                         max_results: int) -> List[dict]:
        """Deduplicate and merge results with diversity scoring"""
        scored_results = []

        # Apply diversity penalty to similar results
        for idx, chunk in enumerate(vector_chunks + kg_chunks):
            similarity_penalty = 0
            # Check similarity with previous results
            for prev in scored_results[:idx]:
                if self._chunk_similarity(chunk['text'], prev['text']) > 0.7:
                    similarity_penalty += diversity_penalty
            scored_results.append({
                **chunk, 'score': chunk['score'] / (1 + similarity_penalty)
                })
        return sorted(scored_results, key=lambda x: x['score'], reverse=True)[:max_results]

    def _chunk_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity score between 0-1"""
        vec1 = self.encoder.encode(text1)
        vec2 = self.encoder.encode(text2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    async def _get_recent_chunks(self, limit=5):
        cypher = """
            MATCH (c:Chunk)
            RETURN c.text AS text, 1.0 as score
            ORDER BY c.created_at DESC
            LIMIT $limit
        """
        async with self.driver.session() as session:
            result = await session.run(cypher, parameters={"limit": limit})
            return await result.data()

    async def _search_entities(self, query: str, score_threshold: float) -> list:
        """Search knowledge graph entities"""
        safe_query = self._sanitize_ft_query(query)
        if not safe_query:
            return []

        cypher = """
            CALL db.index.fulltext.queryNodes("entitySearch", $query)
            YIELD node, score
            WHERE score > $score_threshold
            RETURN node.name AS name, 
                   node.description AS description, 
                   score
            ORDER BY score DESC
            LIMIT 10
        """

        async with self.driver.session() as session:
            result = await session.run(cypher, parameters={
                "query": safe_query, "score_threshold": score_threshold
                })
            return await result.data()

    async def _search_chunks(self, query: str, score_threshold: float) -> list:
        """Search document chunks from PDFs"""
        safe_query = self._sanitize_ft_query(query)
        if not safe_query:
            return []

        cypher = """
            CALL db.index.fulltext.queryNodes("chunkSearch", $query)
            YIELD node, score
            WHERE score > $score_threshold
            RETURN node.text AS text, score
            ORDER BY score DESC
            LIMIT 10
        """

        async with self.driver.session() as session:
            result = await session.run(cypher, parameters={
                "query": safe_query, "score_threshold": score_threshold
                })
            return await result.data()

    @staticmethod
    def _sanitize_ft_query(query: str) -> Optional[str]:
        """Use boolean search mode instead of exact phrase matching"""
        if not query.strip():
            return None

        # Escape special chars but don't wrap in quotes
        escaped = re.sub(r'([\+\-\&\|\!\(\)\{\}\[\]\^\"\~\*\?\:\\/])', r'\\\1', query)

        # Split into terms and join with AND operators for correct full-text bool search
        # --> next stage: add cases or generalize to reduce noise
        terms = [f"*{t}*" for t in escaped.split() if t not in ["please", "summarize", "the", "text"]]
        return " AND ".join(terms) if terms else None

    @staticmethod
    async def _create_entities(tx, data):
        for item in data:
            await tx.run("MERGE (e:Entity {id: $id}) SET e += $props", id=item["name"].lower(), props=item)

    @async_error_handler
    @timing_decorator
    async def ingest_data(self, data_format: str, file_path: str):
        """
        Ingest data from various formats
        Supported formats: CSV, JSON
        """
        logger.info(f"Ingesting {data_format} file: {file_path}")
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"{file_path} does not exist")
        if data_format == "csv":
            await self._ingest_csv(file_path)
        elif data_format == "json":
            await self._ingest_json(file_path)
        else:
            raise ValueError("Unsupported format. Use 'csv' or 'json'")

    async def _ingest_csv(self, file_path: str):
        import pandas as pd
        df = pd.read_csv(file_path)

        async with self.driver.session() as session:
            await session.execute_write(self._process_csv_records, df.to_dict('records'))

    @staticmethod
    async def _process_csv_records(tx, records):
        for record in records:
            await tx.run("""
                MERGE (src:Entity {id: toLower($source)})
                MERGE (tgt:Entity {id: toLower($target)})
                MERGE (src)-[r:RELATES_TO {type: $relationship}]->(tgt)
                """, source=record['source'], target=record['target'], relationship=record['relationship'])

    async def _ingest_json(self, file_path: str):
        import json
        """Process JSON knowledge graph data"""
        with open(file_path, 'r') as f:
            data = json.load(f)

        async with self.driver.session() as session:
            await session.execute_write(self._process_json_data, data)

    @staticmethod
    async def _process_json_data(tx, data):
        for entity in data.get('entities', []):
            await tx.run("MERGE (e:Entity {id: toLower($id)}) SET e += $properties", id=entity['id'],
                         properties=entity['properties'])

        for relationship in data.get('relationships', []):
            await tx.run("""
                MERGE (src:Entity {id: toLower($src_id)})
                MERGE (tgt:Entity {id: toLower($tgt_id)})
                MERGE (src)-[r:${rel_type}]->(tgt)
                SET r += $properties
                """, src_id=relationship['source'], tgt_id=relationship['target'], rel_type=relationship['type'],
                         properties=relationship.get('properties', {}))

    async def ingest_pdf(self, file_path: str, executor: concurrent.futures.ThreadPoolExecutor,
                         original_filename: str) -> str:
        """Process PDF and store chunks in Neo4j + Qdrant"""
        logger.info(f"Starting PDF ingestion: {file_path}")

        try:
            if not os.path.exists(file_path):
                logger.error(f"PDF file not found: {file_path}")
                return f"Error: PDF file not found at {file_path}"

            # Check file size
            file_size = os.path.getsize(file_path)
            logger.info(f"PDF file size: {file_size} bytes")
            if file_size == 0:
                logger.error("PDF file is empty")
                return "Error: PDF file is empty"

            # Extract text from PDF with enhanced extraction
            logger.info("Extracting text from PDF with enhanced methods")
            text = await self._extract_pdf_text(file_path, executor)

            # Check if extraction was successful
            if not text or len(text) < 50:  # Set meaningful minimum threshold
                logger.error("PDF text extraction returned insufficient text")
                return "Error: Could not extract sufficient text from PDF. The file might be corrupted, password-protected, or contain only images."

            logger.info(f"Successfully extracted {len(text)} characters from PDF")
            logger.debug(f"Text preview: {text[:200]}...")

            # Create text chunks
            logger.info(f"Chunking text...")
            logger.info(f"Raw extracted text preview: {text[:500]}")
            chunks = self._chunk_text(text)
            logger.info(f"Created {len(chunks)} text chunks")
            logger.debug(f"First chunk preview: {chunks[0][:200]}...")

            await self._store_chunks(chunks, original_filename)

            entities = await self._extract_entities(text)
            await self._store_entities(entities, original_filename)

            logger.info("PDF ingestion completed successfully")
            return (f"Successfully processed PDF: extracted {len(text)} characters, "
                    f"identified {len(entities)} entities, and created {len(chunks)} knowledge chunks.")

        except Exception as e:
            stack_trace = traceback.format_exc()
            logger.error(f"PDF ingestion failed: {str(e)}\n{stack_trace}")
            return f"Error processing PDF: {str(e)}"

    async def _store_entities(self, entities: List[dict], source: str):
        """Store extracted entities in Neo4j"""
        async with self.driver.session() as session:
            await session.execute_write(self._process_entities, entities, source)

    @staticmethod
    async def _process_entities(tx, entities: List[dict], source: str):
        """statm to process entities before storing"""
        for entity in entities:
            await tx.run("""
                MERGE (e:Entity {id: toLower($text)}) 
                SET e.type = $type,
                    e.source = $source,
                    e.confidence = $confidence,
                    e.last_seen = datetime()
                """, text=entity['text'], type=entity['type'], source=source, confidence=entity['score'])

    async def _store_chunks(self, chunks: List[str], filename: str):
        """Store in both Neo4j and Qdrant"""
        logger.info(f"Storing {len(chunks)} chunks in Neo4j")

        # Generate a unique batch ID for this upload
        batch_id = datetime.now().strftime('%Y%m%d%H%M%S')

        async with self.driver.session() as session:
            # Create document node
            await session.run("CREATE (d:Document {id: $id, name: $name, created_at: datetime()})",
                id=f"doc_{batch_id}", name=filename)

            # Store chunks with relationships
            for i, chunk in enumerate(chunks):
                await session.run("""
                    CREATE (c:Chunk {
                        id: $id,
                        text: $text,
                        source: 'pdf_upload',
                        created_at: datetime()
                    })
                    WITH c
                    MATCH (d:Document {id: $doc_id})
                    CREATE (d)-[:CONTAINS]->(c)
                    """, id=f"chunk_{batch_id}_{i}", text=chunk, doc_id=f"doc_{batch_id}")
        try:
            embeddings = self.encoder.encode(chunks)
            self.qdrant.upsert(
                collection_name="doc_chunks",
                points=[{
                "id": idx,
                "vector": embedding.tolist(),
                "payload": {
                    "text": chunk,
                    "doc_id": f"doc_{batch_id}",
                    "source": filename
                    }
                } for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings))])
            logger.info(f"Stored {len(chunks)} vectors in Qdrant")
        except Exception as e:
            logger.error(f"Failed to store vectors: {str(e)}")
            raise

    async def vector_search(self, query: str, limit=5):
        """Safe vector search with error handling"""
        try:
            if not self.qdrant.collection_exists("doc_chunks"):
                logger.warning("No documents have been ingested yet")
                return []

            query_embedding = self.encoder.encode(query).tolist()
            results = self.qdrant.search(collection_name="doc_chunks", query_vector=query_embedding, limit=limit)
            return [{"text": hit.payload["text"], "score": hit.score} for hit in results]
        except Exception as e:
            logger.error(f"Vector search failed: {str(e)}")
            return []

    async def _extract_pdf_text(self, file_path: str, executor: concurrent.futures.ThreadPoolExecutor) -> str:
        """Extract text from PDF using multiple extraction methods for best results"""
        try:
            logger.info(f"Starting PDF text extraction for {file_path}")
            return await asyncio.get_event_loop().run_in_executor(executor, self._sync_extract_pdf_text, file_path)
        except Exception as e:
            logger.error(f"PDF extraction failed in async wrapper: {str(e)}", exc_info=True)
            raise

    @staticmethod
    async def _extract_entities(text: str) -> List[dict]:
        """Improved named entity recognition"""
        from presidio_analyzer import AnalyzerEngine

        analyzer = AnalyzerEngine()

        results = analyzer.analyze(text=text, language="en")
        entities = []

        for result in results:
            entity_text = text[result.start:result.end]
            entities.append({
                "type": result.entity_type, "text": entity_text, "score": result.score
                })

        # Deduplicate entities
        unique_entities = {e['text']: e for e in entities}.values()
        return list(unique_entities)

    def _sync_extract_pdf_text(self, file_path: str) -> str:
        """Robust PDF text extraction with layout preservation"""
        max_pages = int(os.getenv("PDF_PAGE_LIMIT", "20"))
        extraction_results = []

        # Extracting from PDF depends on content - two methods used to improve extraction
        # Method 1: PyMuPDF
        try:
            import fitz
            text = []
            with fitz.open(file_path) as pdf:
                for i, page in enumerate(pdf):
                    if i >= max_pages:
                        break
                    # Get text blocks in natural reading order
                    blocks = page.get_text("blocks", sort=True)
                    # Filter and join text blocks
                    page_text = " ".join([b[4].strip().replace('\n', ' ') + " " for b in blocks if b[6] == 0])
                    page_text = re.sub(r'\s+', ' ', page_text)
                    text.append(page_text)
            pymupdf_text = " ".join(text)
            extraction_results.append(("PyMuPDF", pymupdf_text))
        except Exception as e:
            logger.error(f"PyMuPDF failed: {str(e)}")

        # Method 2: pdfplumber (table/formatted text)
        try:
            import pdfplumber
            text = []
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    if i >= max_pages:
                        break
                    # Extract text while preserving layout
                    page_text = page.extract_text(x_tolerance=3, y_tolerance=3, keep_blank_chars=False,
                        use_text_flow=True  # Better for formatted docs
                        )
                    # Clean extra spaces between characters
                    page_text = re.sub(r'(?<=\w)\s+(?=\w)', ' ', page_text)
                    text.append(page_text)
            pdfplumber_text = "\n".join(text)
            extraction_results.append(("pdfplumber", pdfplumber_text))
        except Exception as e:
            logger.error(f"pdfplumber failed: {str(e)}")

        # Select best extraction using quality heuristics
        best_text = self._select_best_extraction(extraction_results)
        return self._clean_extracted_text(best_text)

    @staticmethod
    def _select_best_extraction(results):
        """Choose extraction with best word integrity"""

        def quality_score(text):
            # Favor texts with longer average word length
            words = text.split()
            if not words:
                return 0
            avg_len = sum(len(w) for w in words) / len(words)
            return avg_len

        return max(results, key=lambda x: quality_score(x[1]), default=("", ""))[1]

    @staticmethod
    def _clean_extracted_text(text: str) -> str:
        """Improved text cleaning with hyphen handling"""
        # Handle hyphenated words split across lines
        if os.getenv("PDF_HYPHEN_HANDLING", "false").lower() == "true":
            text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2\n', text)

        # Preserve paragraph breaks but clean within paragraphs
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)  # Single newlines to spaces
        text = re.sub(r'\n{3,}', '\n\n', text)  # Limit consecutive newlines
        text = re.sub(r'[^\S\n]+', ' ', text)  # Clean spaces

        return text.strip() # try: text.strip().title()

    @staticmethod
    def _chunk_text(text: str) -> List[str]:
        """Sentence-aware chunking with overlap preservation"""
        max_chunk = int(os.getenv("MAX_TEXT_CHUNK", "4000"))
        overlap = int(os.getenv("TEXT_OVERLAP", "500")) # try different overlaps to reduce info loss

        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0

        # Build chunks respecting sentence boundaries
        for sent in sentences:
            sent_length = len(sent)
            if current_length + sent_length > max_chunk and current_chunk:
                chunks.append(" ".join(current_chunk))
                # Carry over overlap sentences
                overlap_start = max(0, len(current_chunk) - int(overlap / 100 * len(current_chunk)))
                current_chunk = current_chunk[overlap_start:]
                current_length = sum(len(s) for s in current_chunk)

            current_chunk.append(sent)
            current_length += sent_length

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    async def _create_text_chunks(self, chunks: List[str]):
        """Store text chunks in Neo4j with batching for performance"""
        if not chunks:
            logger.warning("No chunks to store in Neo4j")
            return

        try:
            # Generate a unique batch ID for this set of chunks
            batch_id = datetime.now().strftime('%Y%m%d%H%M%S')

            async with self.driver.session() as session:
                # Process chunks in batches
                batch_size = 5  # Process a few chunks per transaction

                for batch_start in range(0, len(chunks), batch_size):
                    batch_end = min(batch_start + batch_size, len(chunks))
                    batch = chunks[batch_start:batch_end]

                    try:
                        # Process this batch in a single transaction
                        await session.execute_write(self._process_chunk_batch, batch, batch_start, batch_id)
                        logger.info(f"Stored chunks {batch_start} to {batch_end - 1}")
                    except Exception as e:
                        logger.error(
                            f"Error storing chunk batch {batch_start}-{batch_end - 1}: {str(e)}")  # Continue with next batch

                # Create relationships between chunks to maintain document flow
                await session.execute_write(self._link_document_chunks, batch_id, len(chunks))

        except Exception as e:
            logger.error(f"Error in Neo4j session while storing chunks: {str(e)}")
            raise

    @staticmethod
    async def _process_chunk_batch(tx, chunks: List[str], start_idx: int, batch_id: str):
        """Store a batch of chunks in a single transaction"""
        for i, chunk in enumerate(chunks):
            chunk_idx = start_idx + i
            await tx.run("""
                CREATE (c:Chunk {
                    id: $id,
                    text: $text,
                    source: 'pdf_upload',
                    index: $index,
                    batch_id: $batch_id,
                    created_at: datetime()
                })
                """, id=f"chunk_{batch_id}_{chunk_idx}", text=chunk, index=chunk_idx, batch_id=batch_id)

    @staticmethod
    async def _link_document_chunks(tx, batch_id: str, chunk_count: int):
        """Create NEXT relationships between chunks to preserve document flow"""
        # This creates a chain of relationships: (chunk0)-[:NEXT]->(chunk1)-[:NEXT]->...
        await tx.run("""
            MATCH (c1:Chunk {batch_id: $batch_id}), (c2:Chunk {batch_id: $batch_id})
            WHERE c1.index = c2.index - 1
            CREATE (c1)-[:NEXT]->(c2)
            """, batch_id=batch_id)

        # Add document relationship to all chunks in this batch
        await tx.run("""
            MATCH (c:Chunk {batch_id: $batch_id})
            MERGE (d:Document {id: $doc_id})
            ON CREATE SET d.created_at = datetime(), d.chunk_count = $count
            CREATE (d)-[:CONTAINS]->(c)
            """, batch_id=batch_id, doc_id=f"doc_{batch_id}", count=chunk_count)

