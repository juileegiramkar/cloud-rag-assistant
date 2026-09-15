import faiss
import numpy as np
import pickle
from pathlib import Path


class VectorStore:
    """
    Stores document embeddings in FAISS and retrieves
    the most semantically similar chunks.
    """

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks = []

    def add(
        self,
        embeddings: list[list[float]],
        chunks: list[dict],
    ):
        """Add embeddings and their corresponding chunks."""

        if len(embeddings) != len(chunks):
            raise ValueError(
                "Embeddings and chunks must have the same length"
            )

        if not embeddings:
            return

        vectors = np.array(
            embeddings,
            dtype="float32",
        )

        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[dict]:
        """Return the most similar document chunks."""

        if not self.chunks:
            return []

        top_k = min(top_k, len(self.chunks))

        query_vector = np.array(
            [query_embedding],
            dtype="float32",
        )

        scores, indices = self.index.search(
            query_vector,
            top_k,
        )

        results = []

        for score, index in zip(scores[0], indices[0]):
            if index == -1:
                continue

            result = self.chunks[index].copy()
            result["score"] = float(score)

            results.append(result)

        return results

    def save(self, directory: str):
        """
        Save the FAISS index and chunk metadata to disk.
        """

        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(
            self.index,
            str(path / "index.faiss"),
        )

        with open(
            path / "chunks.pkl",
            "wb",
        ) as file:
            pickle.dump(self.chunks, file)

    @classmethod
    def load(cls, directory: str):
        """
        Load a previously saved FAISS index and chunk metadata.
        """

        path = Path(directory)

        index = faiss.read_index(
            str(path / "index.faiss")
        )

        with open(
            path / "chunks.pkl",
            "rb",
        ) as file:
            chunks = pickle.load(file)

        store = cls(index.d)
        store.index = index
        store.chunks = chunks

        return store