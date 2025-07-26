# Universität Göttingen: StudIT‑RAG‑Pipeline

Dieses Projekt implementiert eine **Retrieval‑Augmented Generation (RAG)**‑Pipeline mit LangChain, um Fragen rund um die StudIT‑Services der Georg‑August‑Universität Göttingen zu beantworten. Die Lösung besteht aus drei Jupyter‑Notebooks, die zusammen **Web‑Crawling**, **Dokumenten‑Indexierung** und **kontextuelle Antwortgenerierung** mit großen Sprachmodellen durchführen. Entstanden ist das System im Rahmen der Masterarbeit von Jakob Eilts.

---

## Überblick über die Notebooks

| Notebook                           | Zweck                                                                          |
| ---------------------------------- | ------------------------------------------------------------------------------ |
| `1_crawl.ipynb`                    | Crawlt Webseiten und speichert sichtbare Inhalte                               |
| `2_chunk_and_store.ipynb`          | Zerteilt den Text, bettet ihn ein und speichert ihn in einem FAISS‑Vektorindex |
| `3_retrieval_and_generation.ipynb` | Lädt den Index, ruft relevanten Kontext ab und generiert Antworten             |

---

## 1. `1_crawl.ipynb` – Web‑Crawling

Dieses Notebook extrahiert Rohtext von einer vordefinierten Liste von URLs und serialisiert das Ergebnis.

### Schritte

* **Nur sichtbaren Text extrahieren** mit `BeautifulSoup`; entfernt werden `<script>`, `<style>` usw.
* **URLs anfragen und parsen** aus `helper.list_of_all_html.urls`.
* **LangChain‑`Document`‑Objekte** für jede URL erstellen.
* **Liste der `Document`s speichern** als komprimierte Pickle‑Datei: `docs.pkl.gz`.

> **Ergebnis:** `docs.pkl.gz` – GZIP‑komprimierte Liste von LangChain‑Dokumenten inklusive URL‑Metadaten.

---

## 2. `2_chunk_and_store.ipynb` – Text‑Chunking & Indexierung

Dieses Notebook bereitet die Rohdaten für die semantische Suche vor.

### Schritte

#### 1. Dokumente laden

```python
with gzip.open("docs.pkl.gz", "rb") as f:
    docs = pickle.load(f)
```

#### 2. Dokumente zerteilen

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""],
)
chunks = splitter.split_documents(docs)
```

#### 3. Einbettungen erzeugen & Vektorspeicher aufbauen

* Einbettungen werden über die **AcademicCloud‑API der GWDG** erzeugt.
* Indexierung erfolgt mit **FAISS** für schnelle Ähnlichkeitssuche.
* Lokal gespeichert als `faiss_wiki_index`.

```python
store = FAISS.from_documents(chunks, embedder)
store.save_local("faiss_wiki_index")
```

> **Ergebnis:** `faiss_wiki_index/` – Lokaler Vektorspeicher, bereit zum Laden.

---

## 3. `3_retrieval_and_generation.ipynb` – Question Answering

Dieses Notebook orchestriert **Retrieval und Generierung** mit LangChain + LangGraph.

### 1. Prompt definieren

Deutschsprachiger RAG‑Prompt mit Regeln:

* Antworten bevorzugt auf Deutsch
* Kurz, aber präzise
* Bei Unsicherheit vertrauenswürdige Quellen vorschlagen

```text
Du bist der hilfreiche StudIT‑Assistent der Universität Göttingen.
...
Nutze den folgenden Kontext, um die Frage zu beantworten:
{context}
Frage: {question}
Hilfreiche Antwort:
```

### 2. Vektorspeicher & LLM einrichten

* FAISS‑Index laden
* LLM laden: `meta‑llama‑3.1‑8b‑instruct` über die GWDG‑API

```python
vector_store = FAISS.load_local(...)
llm = ChatOpenAI(...)
```

### 3. Graph‑State & Nodes definieren

LangGraph steuert eine 2‑stufige Pipeline:

1. `retrieve(state)`: Semantische Suche basierend auf der Benutzer‑Query.
2. `generate(state)`: Antwort mithilfe des Prompts und des gefundenen Kontexts.

```python
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph = graph_builder.compile()
```

### 4. End‑to‑End‑QA ausführen

```python
question = "Welche Services bietet StudIT?"
result = graph.invoke({"question": question})
print(result["answer"])
```

> **Ausgabe:** Antwort auf Deutsch, abgeleitet aus dem relevanten Dokumentenkontext.

---

## Benötigte Secrets

Die folgenden Secrets sollten sicher in `.streamlit/secrets.toml` (oder einem ähnlichen Mechanismus) abgelegt werden:

```toml
GWDG_API_KEY        = "your-api-key-here"
BASE_URL_EMBEDDINGS = "https://your-embedding-endpoint"
BASE_URL            = "https://your-llm-endpoint"
```

---

## Anforderungen

Installiere Pakete aus der requirements.txt
`pip install -r requirements.txt`

---

## Zusammenfassung

Dieses modulare RAG‑System

* crawlt akademische Webinhalte,
* wandelt Roh‑HTML in vektorisierte Wissenseinheiten um und
* beantwortet Benutzerfragen durch LangChains Orchestrierung.

Für andere Universitäten oder Domänen genügt es, …

* die URL‑Liste in `1_crawl.ipynb` anzupassen und
* das Prompt‑Template in `3_retrieval_and_generation.ipynb` zu verändern.