from sentence_transformers import SentenceTransformer


DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


class Embedder:
    """
    Converts text into dense numerical vectors using a local
    Sentence Transformer model.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Create embeddings for multiple text chunks."""
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Create an embedding for a user query."""
        if not query.strip():
            raise ValueError("query cannot be empty")

        embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        )

        return embedding[0].tolist()