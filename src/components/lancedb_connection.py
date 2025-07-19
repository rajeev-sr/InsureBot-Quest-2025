import os
from dotenv import load_dotenv
import lancedb
import pyarrow as pa
from gemini_model import gemini_embedding


load_dotenv()
def table():
    lancedb_uri=os.getenv("lancedb_uri")
    lancedb_api_key=os.getenv("lancedb_api_key")
    lancedb_region=os.getenv("lancedb_region")

    db = lancedb.connect(
        uri=lancedb_uri,
        api_key=lancedb_api_key,
        region=lancedb_region
    )
    embeddings=gemini_embedding()
    embedding_size = len(embeddings.embed_query("test"))
    if "rag_data" not in db.table_names():
        schema = pa.schema([
            ("id", pa.string()),
            ("text", pa.string()),
            ("embedding", pa.list_(pa.float32(), embedding_size)),
            ("language", pa.string()),
            ("type", pa.string()),
            ("source", pa.string())
        ])
        db.create_table("rag_data", schema=schema)

    table = db.open_table("rag_data")
    return table


if __name__ == "__main__":
    table()