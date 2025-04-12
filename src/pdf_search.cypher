CALL db.index.fulltext.queryNodes("chunkSearch", $query)
YIELD node, score
RETURN node.text AS text, score

MATCH (c:Chunk) WHERE c.source = 'pdf_upload'
RETURN c.text LIMIT 1