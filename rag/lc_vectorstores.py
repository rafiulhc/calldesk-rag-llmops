import os

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

from ingest import EMBEDDING_MODEL, chunk_text, load_documents, split_title

# SQLAlchemy needs the driver in the URL: postgresql+psycopg://...
PG_URL = os.environ["DATABASE_URL"].replace("postgresql://", "postgresql+psycopg://", 1)
QUESTION = "How much do extra minutes cost?"

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)


def show(title, results):
    print(f"\n=== {title} ===")
    for item in results:
        doc, score = item if isinstance(item, tuple) else (item, None)
        label = f"{score:.3f}  " if score is not None else ""
        print(f"  {label}{doc.metadata['source']} #{doc.metadata['chunk_id']}  {doc.page_content.splitlines()[1][:70]!r}")


# Part 1: embeddings
vector = embeddings.embed_query(QUESTION)
print(f"=== Embeddings ===\n  embed_query -> {len(vector)} numbers, first 3: {vector[:3]}")

# Build Documents with YOUR chunker, so results are comparable with your production RAG
docs, ids = [], []
for d in load_documents():
    title, body = split_title(d["text"])
    for i, piece in enumerate(chunk_text(body)):
        docs.append(Document(page_content=f"{title}\n{piece}", metadata={"source": d["source"], "chunk_id": i}))
        ids.append(f"{d['source']}#{i}")
print(f"  {len(docs)} Documents built")

# Part 2: in-memory vector store (no database)
memory_store = InMemoryVectorStore.from_documents(docs, embeddings)
show("In-memory store (score = similarity, higher is better)",
     memory_store.similarity_search_with_score(QUESTION, k=3))

# Part 3: PGVector in your Postgres (its own tables; your 'chunks' table is untouched)
pg_store = PGVector(embeddings=embeddings, collection_name="calldesk_lc", connection=PG_URL, use_jsonb=True)
pg_store.add_documents(docs, ids=ids)     # same ids = update, not duplicate, when you re-run
show("PGVector (score = distance, LOWER is better)",
     pg_store.similarity_search_with_score(QUESTION, k=3))

# Part 4: metadata filtering
refund_q = "Can I get my money back?"
show(f"No filter: {refund_q!r}", pg_store.similarity_search_with_score(refund_q, k=3))
show(f"Filter source = pricing.txt: {refund_q!r}",
     pg_store.similarity_search_with_score(refund_q, k=3, filter={"source": {"$eq": "pricing.txt"}}))

# Part 5: MMR (relevant AND varied)
plan_q = "What does the Growth plan include?"
show(f"Normal search: {plan_q!r}", pg_store.similarity_search(plan_q, k=4))
show(f"MMR (lambda_mult=0.5): {plan_q!r}",
     pg_store.max_marginal_relevance_search(plan_q, k=4, fetch_k=12, lambda_mult=0.5))

# Part 6: a retriever is a Runnable (used in A4's chains)
retriever = pg_store.as_retriever(search_kwargs={"k": 2})
show("Retriever.invoke()", retriever.invoke(QUESTION))
