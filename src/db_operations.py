import asyncio
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


class DBOps:
    def __init__(self):
        self.driver = None
        self.qdrant = QdrantClient(host="qdrant", port=6333)
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

    async def augmented_retrieval(self, query: str, score_threshold: float):
        """Direct hybrid search implementation"""
        # Vector search
        vector_results = await asyncio.get_event_loop().run_in_executor(None,
            lambda: self.qdrant.query_points(collection_name="doc_chunks", query_vector=self.encoder.encode(query).tolist(),
                limit=10))

        # Graph search
        async with self.driver.session() as session:
            kg_results = await session.run("""CALL db.index.fulltext.queryNodes("entitySearch", $query) 
                YIELD node, score WHERE score > $threshold 
                RETURN node.name AS name, node.description AS description, score""",
                {"query": query, "threshold": score_threshold})
            kg_data = await kg_results.data()

        return {
            "vector": [{"text": hit.payload["text"], "score": hit.score} for hit in vector_results], "kg": kg_data
            }