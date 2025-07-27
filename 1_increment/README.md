# **StudIT Chatbot RAG**

Dieses Repository stellt eine **Retrieval‑Augmented‑Generation (RAG)**‑Pipeline bereit, die Fragen zu den IT‑Services der Georg‑August‑Universität Göttingen beantwortet.

| Komponente/Notebook                | Zweck                                                                                                               |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `1_crawl.ipynb`                    | Crawlt Webseiten und speichert sichtbaren Text als `docs.pkl.gz`.                                                   |
| `2_chunk_and_store.ipynb`          | Zerteilt Texte, erstellt Embeddings und legt sie in einem lokalen **FAISS**‑Index (`faiss_wiki_index/`) ab.         |
| `3_retrieval_and_generation.ipynb` | Definiert Prompt, Retrieval‑Nodes und Antwort‑Generierung mithilfe von **LangGraph**.                               |
| `streamlit_app.py`                 | Interaktiver Chatbot auf Basis von **Qwen3‑32B**: RAG‑Antworten, Quellen‑Anzeige, Chat‑Logging und Support‑E‑Mails. |

---

## 1. `1_crawl.ipynb`– Web‑Crawling (unverändert)

* Ruft alle URLs aus `helper.list_of_all_html.urls` ab.
* Entfernt unsichtbare Tags (`<script>`,`<style>` u.a.) in `extract_visible_text`.
* Speichert jede Seite als `langchain.docstore.document.Document` mit `metadata={"url": …}`.
* Serialisiert die Liste gzip‑komprimiert: **`docs.pkl.gz`**.

---

## 2. `2_chunk_and_store.ipynb`– Chunking & FAISS‑Index

1. **Laden** der Rohdokumente aus `docs.pkl.gz`.
2. **Chunking** mit `RecursiveCharacterTextSplitter` (`chunk_size=1000`, `chunk_overlap=200`).
3. **Embeddings** via **GWDG AcademicCloud API** (Konfiguration über `st.secrets`).
4. **Speichern** des FAISS‑Vektorspeichers lokal: **`faiss_wiki_index/`**.

```python
embedder = AcademicCloudEmbeddings(
    api_key=st.secrets["GWDG_API_KEY"],
    url=st.secrets["BASE_URL_EMBEDDINGS"],
)
store = FAISS.from_documents(chunks, embedder)
store.save_local("faiss_wiki_index")
```

---

## 3. `3_retrieval_and_generation.ipynb` – RAG‑Orchestrierung (angepasst)

* **Prompt‑Vorlage** (Deutsch, kurz & zuverlässig) enthält jetzt das Directive‑Token `"/no_think"` und strengere Halluzinations‑Vermeidungsregeln.
* **LLM:** `qwen3‑32b` (Alibaba Qwen3) über die GWDG‑API – ersetzt `meta‑llama‑3.1‑8b‑instruct`.
* **LangGraph‑State** hält Query, Kontext‑Docs und Antworten; **MemorySaver** cacht den Graph‑Zustand.
* **Retrieval‑Tool** liefert jetzt sowohl serialisierten Quelltext **und** die Dokument‑Objekte zurück (`response_format="content_and_artifact"`).
* **Graph**: Query→(optional `retrieve`)→Answer.

```python
# LLM‑Initialisierung (neu)
llm = init_chat_model(
    "qwen3-32b",
    model_provider="openai",
    base_url=st.secrets["BASE_URL"],
    temperature=0.3,
    api_key=st.secrets["GWDG_API_KEY"],
)

@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """FAISS‑Suche (k=2) und Rückgabe der Treffer."""
    docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        f"Source: {d.metadata}\nContent: {d.page_content}" for d in docs
    )
    return serialized, docs
```

---

## 4. `streamlit_app.py` – Chat‑Frontend & Support‑Workflow (neu)

| Bereich               | Änderungen gegenüber der Vorgängerversion                                                                                               |
| --------------------- |-----------------------------------------------------------------------------------------------------------------------------------------|
| **Layout & Branding** | Neues Logo (`st.logo(...)`) und überarbeitete Willkommens‑Nachricht.                                                                    |
| **Chat‑Engine**       | Verwendung von **Qwen3‑32B** via `init_chat_model`; Tool‑Aufrufe über `llm.bind_tools([retrieve])`.                                     |
| **Caching**           | `@st.cache_resource` für die Graph‑Kompilierung – Startzeiten sinken.                                                                   |
| **Graph‑Logik**       | Zwei LLM‑Aufrufe: (1) Tool‑Entscheidung, (2) finale Antwort; Kontext wird als separate AI‑Message injiziert.                            |
| **Quellen‑Anzeige**   | Dynamische Expander mit vollständiger URL als Titel.                                                                                    |
| **Support‑Ticket**    | Formular in separater Funktion `support_form()`. HTML‑E‑Mails mit gebrandetem Kopfbereich; verbesserte Validierung und Fehlermeldungen. |
| **Logging**           | Unverändert: JSONL‑Protokoll unter `chat_logs/`.                                                                                        |

### Schnelle Code‑Highlights

**Willkommens‑Nachricht & Logo**

```python
st.set_page_config(page_title="StudIT‑Chatbot", page_icon="💬", layout="centered")
st.title("StudIT‑Chatbot")
st.logo("helper/images/uni_logo.png")
```

**Graph‑Caching**

```python
@st.cache_resource(show_spinner="Initialisiere Modelle…")
def build_graph():
    ...
```

**Support‑E‑Mail‑Versand** (unverändert, jetzt mit HTML‑Variante)

```python
def send_support_mail(subject, plain_body, html_body=None):
    cfg = st.secrets["EMAIL"]
    ...
```

---

## Benötigte **Secrets** (unverändert)

```toml
# .streamlit/secrets.toml -----------------------------
GWDG_API_KEY        = "…"
BASE_URL_EMBEDDINGS = "https://embedding-endpoint"
BASE_URL            = "https://llm-endpoint"

[EMAIL]
SMTP_SERVER = "mail.uni-goettingen.de"
SMTP_PORT   = "465"          # 465 = SMTPS, 587 = STARTTLS
SMTP_USER   = "chatbot@uni-goettingen.de"
SMTP_PASS   = "…"
SUPPORT_TO  = "studIT-support@uni-goettingen.de"
```

---

## Installation & Start

```bash
# Abhängigkeiten
pip install -r requirements.txt

# Crawl → Index → Start App
jupyter nbconvert --execute 1_crawl.ipynb
jupyter nbconvert --execute 2_chunk_and_store.ipynb
streamlit run streamlit_app.py
```

> **Tipp:** Die Notebooks 1 & 2 müssen nur erneut ausgeführt werden, wenn sich die Quell‑Webseiten ändern.

---

## Zusammenfassung

Die aktuelle StudIT‑RAG‑Pipeline (Release 2025‑07)

* **crawlt**, **indexiert** und **beantwortet** Fragen zu IT‑Services der Universität Göttingen,
* nutzt das leistungsstarke **Qwen3‑32B‑Modell** für präzisere Antworten,
* bietet ein **komfortables Chat‑Frontend** mit Quellen‑Transparenz und gebrandetem Design,
* **loggt** sämtliche Unterhaltungen vollautomatisch,
* und ermöglicht bei Bedarf den direkten **Support‑Kontakt per E‑Mail** – inklusive strukturiertem HTML‑Layout und Chat‑Zusammenfassung.

Durch Anpassung der URL‑Liste bzw. des Prompts lässt sich das System weiterhin problemlos auf andere Einrichtungen übertragen.
