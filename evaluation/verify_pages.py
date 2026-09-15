import json

from src.ingestion.embedder import Embedder
from src.rag.vector_store import VectorStore


# Load evaluation dataset.
with open(
    "evaluation/eval_dataset.json",
    "r",
    encoding="utf-8",
) as file:
    dataset = json.load(file)


# Load the existing vector store.
vector_store = VectorStore.load(
    "data/processed/vector_store"
)

embedder = Embedder()


for item in dataset:
    question = item["question"]

    print("\n" + "=" * 70)
    print(f"Question: {question}")
    print(f"Expected topic: {item['expected_topic']}")
    print(f"Expected relevant pages: {item['relevant_pages']}")

    query_embedding = embedder.embed_query(question)

    results = vector_store.search(
        query_embedding,
        top_k=5,
    )

    print("\nRetrieved pages:")

    for result in results:
        print(
            f"Page {result['page']} | "
            f"Score: {result['score']:.4f}"
        )

        print(
            f"Text: {result['text'][:250].replace(chr(10), ' ')}..."
        )