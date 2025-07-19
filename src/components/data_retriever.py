from langchain_core.retrievers import BaseRetriever
from pydantic import Field
from typing import Any
from typing import List
from langchain_core.documents import Document
import lancedb

from lancedb_connection import table
from gemini_model import gemini_embedding



class LanceDBRemoteRetriever(BaseRetriever):
    table: Any = Field(exclude=True)
    embedding_model: Any = Field(exclude=True)
    k: int = 3
    vector_key: str = "embedding"

    def _get_relevant_documents(self, query: str) -> List[Document]:
        query_vector = self.embedding_model.embed_query(query)
        results = (
            self.table.search(query_vector, vector_column_name=self.vector_key)
            .limit(self.k)
            .to_arrow()
        )

        result_dicts = results.to_pydict() 

        docs = [
            Document(
                page_content=text,
                metadata={
                    "source": source or "unknown",
                    "language": language or "unknown",
                    "type": doc_type or "unknown"
                }
            )
            for text, source, language, doc_type in zip(
                result_dicts["text"],
                result_dicts.get("source", ["unknown"] * len(result_dicts["text"])),
                result_dicts.get("language", ["unknown"] * len(result_dicts["text"])),
                result_dicts.get("type", ["unknown"] * len(result_dicts["text"]))
            )
        ]

        return docs

    def invoke(self, query: str, **kwargs) -> List[Document]:
        return self._get_relevant_documents(query)



def get_rag_response(user_query):
    tb=table()
    embedding_function=gemini_embedding()
    retriever = LanceDBRemoteRetriever(
        table=tb,
        embedding_model=embedding_function,
        k=5
    )

    result = retriever.invoke(user_query)
    return result
