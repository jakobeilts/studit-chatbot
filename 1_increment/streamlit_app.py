import os
import json
import uuid
import pathlib
from datetime import datetime, timezone

import streamlit as st
from langchain.chat_models import init_chat_model
from langchain_community.vectorstores import FAISS
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, AIMessage
from langchain_core.documents import Document
from typing_extensions import List
import smtplib, ssl, html
from email.message import EmailMessage
from helper.academicCloudEmbeddings import AcademicCloudEmbeddings
st.set_page_config(page_title="StudIT‑Chatbot", page_icon="💬", layout="centered")

# ----------------------------------------------------------------------
# 1.   Logging‑Utility (ein JSONL‑Eintrag pro Chat‑Nachricht)
# ----------------------------------------------------------------------
LOG_DIR = "chat_logs"
pathlib.Path(LOG_DIR).mkdir(exist_ok=True)

if "chat_id" not in st.session_state:   # UUID je Browser‑Session
    st.session_state.chat_id = str(uuid.uuid4())

def write_log(role: str, content: str) -> None:
    """Hängt eine Zeile {ts, role, content} an die Log‑Datei der Session an."""
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "role": role,
        "content": content,
    }
    log_file = os.path.join(LOG_DIR, f"{st.session_state.chat_id}.jsonl")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# ----------------------------------------------------------------------
# 2.   Prompt‑Vorlage für RAG‑Antworten
# ----------------------------------------------------------------------
BASE_RULES = (
    "/no_think\n"
    "Du bist der hilfreiche StudIT‑Assistent der Universität Göttingen.\n"
    "• Antworte bevorzugt auf Deutsch.\n"
    "• Antworte so kurz wie möglich, aber so ausführlich wie nötig.\n"
    "• Stelle IMMER eine Rückfrage, wenn wichtige Informationen fehlen – z.B. das Betriebssystem, der Standort oder die Zielgruppe. Mache KEINE Annahmen.\n"
    "• Falls du eine fachliche Frage beantwortest, RUFE IMMER zuerst das Tool `retrieve` auf und beantworte erst dann. Antworte niemals mit Fakten aus anderen Quellen.\n"
    "• Falls keine Info da ist, verweise auf offizielle Kontakte (Link/E-Mail).\n"
    "• Alle Fragen beziehen sich auf die Georg-August-Universität Göttingen.\n"
    "HALLUZINIERE NICHT.\n"
)

# ----------------------------------------------------------------------
# 3.   Vektor‑Datenbank + LLM‑Instanz
# ----------------------------------------------------------------------
VECTORSTORE_PATH = "faiss_wiki_index"

embedder = AcademicCloudEmbeddings(
    api_key=st.secrets["GWDG_API_KEY"],
    url=st.secrets["BASE_URL_EMBEDDINGS"],
)

vector_store = FAISS.load_local(
    VECTORSTORE_PATH,
    embedder,
    allow_dangerous_deserialization=True,
)

llm = init_chat_model(
    "qwen3-32b",
    model_provider="openai",
    base_url=st.secrets["BASE_URL"],
    temperature=0.3,
    api_key=st.secrets["GWDG_API_KEY"],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)

# ----------------------------------------------------------------------
# 4.   LangGraph‑State + Retrieval‑Tool
# ----------------------------------------------------------------------
class State(MessagesState):
    context: List[Document]          # für spätere Anzeige der Quellen
    asked: bool = False  # ob eine Rückfrage gestellt wurde
    current_tool_docs: List[Document] = []

@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Sucht passende Wiki‑Dokumente per FAISS und liefert sie zurück."""
    docs = vector_store.similarity_search(query, k=4)
    print(f"\n📄 RETRIEVED CONTEXTS  (k={len(docs)})")
    for i, d in enumerate(docs, 1):
        print(f"\n─── DOC {i} ─────────────────────────────────")
        print("metadata:", json.dumps(d.metadata, ensure_ascii=False, indent=2))
        print("content:\n", d.page_content)
    serialized = "\n\n".join(
        (f"Source: {d.metadata}\nContent: {d.page_content}") for d in docs
    )
    return serialized, docs

@tool(response_format="content")
def ask(missing_info: str) -> str:
    """Stellt eine gezielte Rückfrage an den Nutzer, wenn Informationen fehlen."""
    return f"Ich brauche noch folgende Information, um weiterzumachen: {missing_info}"


def query_or_respond(state: State):
    """Erster LLM-Aufruf: prüft, ob ein Tool nötig ist."""
    messages = [SystemMessage(BASE_RULES)] + state["messages"]
    resp = llm.bind_tools([retrieve, ask]).invoke(messages)
    return {"messages": [resp]}



tools = ToolNode([retrieve, ask])


def generate(state: State):
    """Zweiter LLM-Aufruf: nutzt Kontext und gibt finale Antwort."""
    # 1. Dokumente aus Tools sammeln
    recent_tool_msgs = [m for m in reversed(state["messages"]) if m.type == "tool"]
    ctx_blocks = "\n\n".join(m.content for m in recent_tool_msgs)

    # 2. Basis-Prompt
    messages = [SystemMessage(BASE_RULES)]

    # 3. Kontext als separate Assistant-Message hinzufügen
    # 3. Kontext als separate Assistant‑Message hinzufügen
    if ctx_blocks:
        messages.append(
            AIMessage(content=f"Verwende diese kontextuellen Abschnitte:\n\n{ctx_blocks}")
        )

    # 4. Gesprächsverlauf (Human + AI‑Turns OHNE Tool‑Calls)
    for m in state["messages"]:
        if m.type == "human":
            messages.append(m)
        elif m.type == "ai" and not getattr(m, "tool_calls", None):
            messages.append(m)

    # 5. Finaler Aufruf
    resp = llm.invoke(messages)

    # 6. Dokumente speichern
    ctx_docs = []
    for tm in reversed(state["messages"]):
        if tm.type == "tool":
            artifacts = getattr(tm, "artifact", None)
            if artifacts:
                ctx_docs.extend(artifacts)
            break  # Nur den letzten Tool-Call verwenden

    return {"messages": [resp], "context": ctx_docs, "current_tool_docs": ctx_docs}


# ----------------------------------------------------------------------
# 5.   Graph‑Definition (Query → ggf. Tool → Answer)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Initialisiere Modelle…")
def build_graph():
    g = StateGraph(State)
    g.add_node(query_or_respond)
    g.add_node(tools)

    def tool_decider(state: State):
        tool_msg = [m for m in state["messages"] if m.type == "tool"]
        if not tool_msg:
            return END

        last_tool = tool_msg[-1].name

        if last_tool == "retrieve":
            return "generate"

        elif last_tool == "ask":
            if state.get("asked"):  # Schon gefragt → abbrechen
                return END
            else:
                state["asked"] = True
                return END  # War erste Frage → auf Antwort warten
        return END

    g.add_node(generate)

    g.set_entry_point("query_or_respond")
    g.add_conditional_edges(
        "query_or_respond",
        tools_condition,
        {
            END: END,
            "tools": "tools",
        },
    )
    g.add_conditional_edges("tools", tool_decider, {"generate": "generate", END: END})
    g.add_edge("generate", END)

    return g.compile(checkpointer=MemorySaver())  # MemorySaver = Zustands‑Cache

graph = build_graph()
config = {"configurable": {"thread_id": "abc123"}}

# ----------------------------------------------------------------------
# 6.   Streamlit‑UI
# ----------------------------------------------------------------------
st.title("StudIT‑Chatbot")
st.logo("helper/images/uni_logo.png")

# ----------------------------------------------------------
# Einmalige Willkommens‑Nachricht (nicht im Verlauf speichern)
# ----------------------------------------------------------
with st.chat_message("assistant"):
    st.markdown(
        "Hallo und willkommen beim **StudIT‑Chatbot**! "
        "Stell mir gerne deine Frage rund um IT‑Services an der Universität Göttingen."
        "Sollte ich eine Frage nicht beantworten können oder deine Anfrage noch mehr Service benötigt als ich leisten kann, kannst du auch direkt über den \"Support-Ticket öffnen\" Button ein Ticket an den Support schreiben."
    )

# Session‑State‑Initialisierung
for key, default in {
    "messages": [],
    "support_form_visible": False,
    "support_summary": "",
    "support_full_chat": "",
}.items():
    st.session_state.setdefault(key, default)

# Bisherige Chat‑Nachrichten anzeigen
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ----------------------------------------------------------
# Nutzer‑Eingabe → Graph → Antwort
# ----------------------------------------------------------
if user_input := st.chat_input("Frag mich etwas …",  key="main_chat"):
    print(f"\n🟢 USER → {user_input}")
    # 1) Nutzer‑Nachricht speichern + loggen
    st.session_state.messages.append({"role": "user", "content": user_input})
    write_log("user", user_input)
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2) Graph ausführen
    with st.spinner("Denke nach…"):
        result = graph.invoke({"messages": st.session_state.messages}, config=config)

    # 3) Antwort speichern + loggen
    ai_msg = result["messages"][-1].content
    print("\n🤖 BOT ←\n", ai_msg)
    st.session_state.messages.append({"role": "assistant", "content": ai_msg})
    write_log("assistant", ai_msg)
    with st.chat_message("assistant"):
        st.markdown(ai_msg)

        ######################################################################
        #  ### NEW CODE: build & store the compact RAG report  ###############
        ######################################################################
        # Gather up to 4 retrieved passages (already limited to k=4)
        retrieved_contexts = [
            doc.page_content if hasattr(doc, "page_content") else doc.get("page_content", "")
            for doc in result.get("current_tool_docs", [])
        ]

        # Build the JSON object
        rag_report = {
            "user_input": user_input,  # already have it
            "reference": "",  # fill in if you have a gold answer
            "retrieved_contexts": retrieved_contexts,
            "response": ai_msg,
        }

        # Decide what you want to do with it – here we append to a second log file
        rag_log_file = os.path.join(LOG_DIR, f"EVAL_{st.session_state.chat_id}_rag.jsonl")
        print("► Saving RAG log…")
        print(json.dumps(rag_report, ensure_ascii=False, indent=2))
        ######################################################################

    # 4) Quellen‑Expander
    if result.get("current_tool_docs"):
        st.markdown("### Quellen & Ausschnitte")

        seen_sources = set()

        for doc in result["current_tool_docs"]:
            meta = doc.metadata if hasattr(doc, "metadata") else doc.get("metadata", {})
            text = doc.page_content if hasattr(doc, "page_content") else doc.get("page_content", "")
            source = meta.get("url") or meta.get("source") or "Unbekannte Quelle"

            if source in seen_sources:
                continue
            seen_sources.add(source)

            with st.expander(source):
                st.markdown(f"[Zur Quelle]({source})")
                st.markdown(f"> {text[:400]}…")

# ----------------------------------------------------------
# Support‑Ticket‑Button (immer sichtbar)
# ----------------------------------------------------------
if st.button("Support‑Ticket öffnen"):
    # Vollständigen Chat dumpen
    chat_dump = "\n".join(f"{m['role'].title()}: {m['content']}"
                          for m in st.session_state.messages)
    st.session_state.support_full_chat = chat_dump

    # 2‑3‑Satz‑Summary via LLM (oder leer, falls noch kein Chat)
    if st.session_state.messages:
        summary_prompt = (
            "Fasse den folgenden Dialog zwischen einem Nutzer und dem StudIT‑Chatbot "
            "in 2‑3 Sätzen zusammen:\n\n"
            f"{chat_dump}"
        )
        st.session_state.support_summary = llm.invoke(summary_prompt).content.strip()
    else:
        st.session_state.support_summary = ""

    st.session_state.support_form_visible = True

# ----------------------------------------------------------------------
# E-Mail Versand
# ----------------------------------------------------------------------
def send_support_mail(subject: str, plain_body: str, html_body: str | None = None) -> None:
    """Versendet Plain‑Text + optional HTML an SUPPORT_TO."""
    cfg = st.secrets["EMAIL"]

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"]    = cfg["SMTP_USER"]
    msg["To"]      = cfg["SUPPORT_TO"]
    msg.set_content(plain_body)                 # Fallback‑Teil

    if html_body:                               # hübsche Variante
        msg.add_alternative(html_body, subtype="html")

    server, port = cfg["SMTP_SERVER"], int(cfg["SMTP_PORT"])

    if port == 465:  # SMTPS
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(server, port, context=context) as smtp:
            smtp.login(cfg["SMTP_USER"], cfg["SMTP_PASS"])
            smtp.send_message(msg)
    else:            # STARTTLS (Port 587 bei WEB.DE zum Beispiel)
        context = ssl.create_default_context()
        with smtplib.SMTP(server, port) as smtp:
            smtp.starttls(context=context)
            smtp.login(cfg["SMTP_USER"], cfg["SMTP_PASS"])
            smtp.send_message(msg)


# ----------------------------------------------------------------------
# Support‑Formular (wird per State‑Flag angezeigt)
# ----------------------------------------------------------------------
def support_form() -> None:
    st.subheader("Support‑Anfrage")

    name  = st.text_input("Name",  value=st.session_state.get("support_name",  ""))
    email = st.text_input("E‑Mail", value=st.session_state.get("support_email", ""))

    chat_exists = bool(st.session_state.messages)
    addition_label = "Anliegen" if not chat_exists else "Optional: Ergänze dein Anliegen"
    addition = st.text_area(addition_label, height=100)

    if st.button("Supportanfrage absenden"):
        # Basis‑Validierung
        errors = []
        if not name.strip():  errors.append("Name darf nicht leer sein.")
        if not email.strip(): errors.append("E‑Mail‑Adresse darf nicht leer sein.")
        if not chat_exists and not addition.strip():
            errors.append("Bitte gib dein Anliegen ein.")
        if errors:
            for err in errors:
                st.error(err)
            return

        # Mail‑Preview zusammenbauen
        chat_dump = "\n".join(f"{m['role'].title()}: {m['content']}"
                              for m in st.session_state.messages)
        summary = st.session_state.support_summary or addition.strip()

        full_msg = (
            "Hallo StudIT‑Team,\n\n"
            "folgende Anfrage stammt aus dem FAQ‑Chatbot:\n\n"
            f"{summary}\n\n\n"
            "Zusätzliche Nachricht des Nutzers/der Nutzerin:\n\n"
            f"{addition.strip()}\n\n\n"
            "Vollständiger Chatverlauf:\n"
            f"{chat_dump}"
        )

        # … innerhalb der support_form(), direkt vor send_support_mail():
        plain_msg = full_msg  # schon vorbereitet

        # Zeilenumbrüche → <br>, bevor man in den f‑String geht
        summary_html = html.escape(summary).replace("\n", "<br>")
        addition_html = html.escape(addition).replace("\n", "<br>")
        chat_html = html.escape(chat_dump)  # pre‑escaped; <pre> behält \n

        html_msg = f"""
        <!DOCTYPE html>
        <html>
          <body style="font-family:sans-serif; line-height:1.5;">
            <h2 style="color:#1a4180;">StudIT‑Supportanfrage</h2>

            <p><strong>Von:</strong> {html.escape(name)} &lt;{html.escape(email)}&gt;</p>

            <h3 style="margin-bottom:4px;">Kurzzusammenfassung</h3>
            <p>{summary_html}</p>

            <h3 style="margin-bottom:4px;">Zusätzliche Nachricht</h3>
            <p>{addition_html or "- (keine) -"}</p>

            <h3 style="margin-bottom:4px;">Vollständiger Chat</h3>
            <pre style="
                 background:#f4f6fa;
                 padding:10px 14px;
                 border:1px solid #dce0e6;
                 border-radius:6px;
                 white-space:pre-wrap;
                 font-family:SFMono-Regular,Consolas,monospace;
            ">{chat_html}</pre>
          </body>
        </html>
        """

        try:
            send_support_mail("Chatbot‑Supportanfrage", plain_msg, html_msg)
            st.info("E‑Mail wurde versendet.")
        except Exception as e:
            st.error(f"E‑Mail‑Versand fehlgeschlagen: {e}")

        st.success("Support‑Anfrage vorbereitet (E‑Mail‑Versand folgt).")
        st.markdown("### Vorschau")
        st.markdown(f"**Name:** {name}")
        st.markdown(f"**E‑Mail:** {email}")
        st.markdown("**Nachricht an StudIT:**")
        st.markdown(full_msg)

        # Persistente Werte speichern & Formular schließen
        st.session_state.update(
            support_form_visible=False,
            support_name=name,
            support_email=email,
            support_addition=addition,
        )

if st.session_state.get("support_form_visible"):
    support_form()
