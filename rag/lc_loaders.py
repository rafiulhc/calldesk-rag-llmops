import os
from pathlib import Path

os.environ.setdefault("USER_AGENT", "llmops-starter-learning/0.1")   # identifies us politely to websites

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingest import chunk_text, load_documents, split_title

DATA_DIR = Path(__file__).parent / "data"
PDF_PATH = Path(__file__).parent / "raw" / "pricing.pdf"


def show(title, docs, n=2):
    print(f"\n=== {title}: {len(docs)} documents ===")
    for d in docs[:n]:
        print("  metadata:", d.metadata)
        print("  content: ", repr(d.page_content[:150]))


def length_stats(texts):
    lengths = [len(t) for t in texts]
    return f"min {min(lengths)}, avg {sum(lengths) // len(lengths)}, max {max(lengths)}"


def starts_mid_sentence(text):
    return not (text[:1].isupper() or text[:1].isdigit())


# Part 1: load the text files
txt_docs = DirectoryLoader(str(DATA_DIR), glob="*.txt", loader_cls=TextLoader).load()
show("Text files", txt_docs)

# Part 2: load a live web page
web_docs = WebBaseLoader("https://calldesk.app/pricing").load()
show("Web page", web_docs, n=1)
print("  total characters:", len(web_docs[0].page_content))

# Part 3: load a PDF (one Document per page)
if PDF_PATH.exists():
    pdf_docs = PyPDFLoader(str(PDF_PATH)).load()
    show("PDF", pdf_docs)
else:
    print(f"\n=== PDF: skipped (save calldesk.app/pricing as {PDF_PATH} to try it) ===")

# Part 4: split with LangChain's recursive splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100, add_start_index=True)
lc_chunks = splitter.split_documents(txt_docs)
show("LangChain chunks", lc_chunks, n=3)

# Part 5: compare with your own chunker
mine = [c for d in load_documents() for c in chunk_text(split_title(d["text"])[1])]
theirs = [c.page_content for c in lc_chunks]
print("\n=== Comparison (same 3 files) ===")
print(f"  my chunker:   {len(mine):>3} chunks  ({length_stats(mine)})  mid-sentence starts: {sum(map(starts_mid_sentence, mine))}")
print(f"  LangChain:    {len(theirs):>3} chunks  ({length_stats(theirs)})  mid-sentence starts: {sum(map(starts_mid_sentence, theirs))}")