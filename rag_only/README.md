# Universität Göttingen: StudIT RAG Pipeline

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline using LangChain to provide question-answering support about StudIT services at Geor-August-Universität Göttingen. The system involves three Jupyter Notebooks that together perform **web crawling**, **document indexing**, and **contextual answer generation** using large language models. It was created as part of the Master Thesis of Jakob Eilts.

---

## Notebooks Overview

| Notebook                           | Purpose                                                            |
| ---------------------------------- | ------------------------------------------------------------------ |
| `1_crawl.ipynb`                    | Crawls web pages and stores visible content                        |
| `2_chunk_and_store.ipynb`          | Chunks text, embeds it, and stores it in a FAISS vector index      |
| `3_retrieval_and_generation.ipynb` | Loads the index, retrieves relevant context, and generates answers |

---

## 1. `1_crawl.ipynb` – Web Crawling

This notebook scrapes raw text content from a predefined list of URLs and serializes the output.

### Steps:

* **Extract visible text only** using `BeautifulSoup`, removing `<script>`, `<style>`, etc.
* **Request and parse URLs** in `helper.list_of_all_html.urls`.
* **Create LangChain `Document` objects** for each URL.
* **Save the list of `Document`s** to a compressed pickle file: `docs.pkl.gz`.

> Output: `docs.pkl.gz` – A GZIP-compressed list of LangChain documents with URL metadata.

---

## 2. `2_chunk_and_store.ipynb` – Text Chunking & Indexing

This notebook prepares the raw data for semantic search.

### Steps:

#### 1. Load Documents

Loads the previously crawled documents:

```python
with gzip.open("docs.pkl.gz", "rb") as f:
    docs = pickle.load(f)
```

#### 2. Chunk Documents

Splits the text into overlapping chunks:

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""],
)
chunks = splitter.split_documents(docs)
```

#### 3. Generate Embeddings & Build Vector Store

* Embeddings generated using **GWDG’s AcademicCloud API**.
* Indexed with **FAISS** for fast similarity search.
* Saved locally as `faiss_wiki_index`.

```python
store = FAISS.from_documents(chunks, embedder)
store.save_local("faiss_wiki_index")
```

> Output: `faiss_wiki_index/` – Local vector store folder ready for loading.

---

## 3. `3_retrieval_and_generation.ipynb` – Question Answering

This notebook implements the **retrieval and generation orchestration** using LangChain + LangGraph.

### 1. Define Prompt

Custom RAG prompt in German with rules:

* Prefer German responses
* Be brief but precise
* Suggest trusted sources when unsure

```text
Du bist der hilfreiche StudIT‑Assistent der Universität Göttingen.
...
Nutze den folgenden Kontext um die Frage zu beantworten:
{context}
Frage: {question}
Helpful Answer:
```

### 2. Setup Vector Store & LLM

* Loads FAISS index
* Loads LLM: `meta-llama-3.1-8b-instruct` from GWDG via API

```python
vector_store = FAISS.load_local(...)
llm = ChatOpenAI(...)
```

### 3. Define Graph State & Nodes

LangGraph orchestrates a 2-step pipeline:

1. `retrieve(state)`: Semantic search based on user query.
2. `generate(state)`: Answer using prompt and retrieved context.

```python
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph = graph_builder.compile()
```

### 4. Run End-to-End QA

Define your question and run the graph:

```python
question = "Welche Services bietet studIT?"
result = graph.invoke({"question": question})
print(result["answer"])
```

> Output: Answer in German, derived from the relevant document context.

---

## Secrets Required

Store the following secrets securely in `.streamlit/secrets.toml` or a similar mechanism used by Streamlit:

```toml
GWDG_API_KEY = "your-api-key-here"
BASE_URL_EMBEDDINGS = "https://your-embedding-endpoint"
BASE_URL = "https://your-llm-endpoint"
```

---

## Requirements

Ensure the following packages are installed:

* `langchain`
* `langchain_community`
* `langgraph`
* `streamlit`
* `faiss-cpu` or `faiss-gpu`
* `beautifulsoup4`, `requests`, `pickle`, `gzip`

---

## Summary

This modular RAG system:

* Crawls and stores academic web content
* Converts raw HTML into vectorized knowledge
* Answers user queries using LangChain's orchestration

The system can be adapted to other universities or domains by updating:

* The `urls` list (in `1_crawl.ipynb`)
* Prompt template (in `3_retrieval_and_generation.ipynb`)

---

**Ready to deploy your own contextual assistant!**

---

**a.** Add unit tests for crawling, chunking, and retrieval
**b.** Add a web UI using Streamlit to ask questions interactively
