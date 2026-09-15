from src.ingestion.embedder import Embedder
from src.rag.vector_store import VectorStore
from src.rag.llm import LocalLLM


class RAGPipeline:
    """
    Combines query embedding, vector retrieval,
    and local LLM generation into a RAG pipeline.
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        llm: LocalLLM,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm = llm

    def answer(
        self,
        question: str,
        top_k: int = 5,
    ) -> dict:
        """
        Retrieve relevant document chunks and generate
        an answer using the local LLM.
        """

        if not question.strip():
            raise ValueError("question cannot be empty")

        # 1. Convert the question into an embedding.
        query_embedding = self.embedder.embed_query(question)

        # 2. Retrieve the most relevant document chunks.
        results = self.vector_store.search(
            query_embedding,
            top_k=top_k,
        )

        if not results:
            return {
                "answer": "I could not find relevant information in the documentation.",
                "sources": [],
            }

        # 3. Reject results that are not sufficiently relevant.
        MIN_SCORE = 0.40

        relevant_results = [
            result
            for result in results
            if result["score"] >= MIN_SCORE
        ]

        if not relevant_results:
            return {
                "answer": (
                    "I could not find enough relevant information "
                    "in the documentation to answer this question."
                ),
                "sources": [],
            }

        results = relevant_results

        # 4. Build context from retrieved chunks.
        context_parts = []

        for i, result in enumerate(results, start=1):
            context_parts.append(
                f"[Source {i} | Page {result['page']}]\n"
                f"{result['text']}"
            )

        context = "\n\n".join(context_parts)

        # 5. Create a grounded prompt.
        prompt = f"""
You are an AI assistant that answers questions using
the provided Amazon ECS documentation.

Use ONLY the information in the provided context.

Rules:
1. Do not use information that is not supported by the context.
2. If the context does not contain enough information,
   say that the documentation does not provide enough information.
3. When making a factual claim, cite the supporting source
   using its source number, for example [Source 1].
4. Do not invent source numbers.

Question:
{question}

Context:
{context}

Answer:
"""

        # 6. Generate the answer with the local LLM.
        answer = self.llm.generate(prompt)

        # 7. Return the answer and source metadata.
        sources = [
            {
                "page": result["page"],
                "source": result["source"],
                "chunk_id": result["chunk_id"],
                "score": result["score"],
            }
            for result in results
        ]

        return {
            "answer": answer,
            "sources": sources,
        }