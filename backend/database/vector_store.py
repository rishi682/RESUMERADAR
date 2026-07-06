import chromadb
from chromadb.utils import embedding_functions

_CHROMA_PATH = "data/chroma"

_client = chromadb.PersistentClient(path=_CHROMA_PATH)

_embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

_collection = _client.get_or_create_collection(
    name="resumes",
    embedding_function=_embedding_function,
)


def add_document(doc_id: str, text: str, metadata: dict | None = None) -> None:
    """Add or update a single document in the vector store."""
    kwargs = {"ids": [doc_id], "documents": [text]}
    if metadata:
        kwargs["metadatas"] = [metadata]
    _collection.upsert(**kwargs)


def query(text: str, n_results: int = 5) -> dict:
    """Query the vector store for the most semantically similar documents."""
    return _collection.query(
        query_texts=[text],
        n_results=n_results,
    )


def delete_document(doc_id: str) -> None:
    """Delete a document from the vector store by id."""
    _collection.delete(ids=[doc_id])


def count() -> int:
    """Return the number of documents currently stored."""
    return _collection.count()