import os
import json
import pickle
import uuid
import pathlib
from datetime import datetime, timezone
from textwrap import dedent

import streamlit as st
from langchain.chat_models import init_chat_model
from langchain.retrievers import ParentDocumentRetriever
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
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

BASE = pathlib.Path(__file__).parent.resolve()
VECTORSTORE_DIR = BASE / "faiss_child_index"
PARENT_STORE_PATH = BASE / "parent_store.pkl"

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
BASE_RULES = dedent("""/no_think
    Du bist der hilfreiche StudIT‑Assistent der Universität Göttingen.
    • Antworte bevorzugt auf Deutsch.
    • Antworte so kurz wie möglich, aber so ausführlich wie nötig.
    • Stelle gezielte Rückfragen, wenn Informationen fehlen, die für eine passende Auswahl oder Handlung unbedingt notwendig sind (z.B. Betriebssystem, Nutzergruppe, Standort).
    • Stelle keine Rückfragen, wenn der abgerufene Kontext bereits eindeutig beantwortet werden kann.
    • Stelle keine hypothetischen oder spekulativen Rückfragen – orientiere dich nur an den tatsächlichen Inhalten.
    Was kann in der RAG Datenbank gesucht werden?
    IT- und studIT-Services
    Supportangebote, Beratungszeiten, Standorte, Kontaktmöglichkeiten zu studIT (IT-Service für Studierende und Mitarbeitende).
    Historie von studIT und Entwicklung der IT-Dienste für Studierende.
    Studierendenaccount & Zugangssysteme
    Alles rund um den Studierendenaccount: Erstellung, Rechte, Passwortverwaltung, Multifaktor-Authentifizierung, Namensänderung, Zusammenlegung mit Vorkurs-Accounts, Zugang nach Exmatrikulation.
    Probleme mit dem Account (z.B. Sperrung, Loginprobleme) und deren Lösung.
    Nutzungsordnung, Rechte und Pflichten, Datenschutz, rechtliche Rahmenbedingungen.
    E-Mail & Kommunikation
    Anleitungen und Einstellungen zur Nutzung des studentischen E-Mail-Postfachs (Outlook, Thunderbird, Apple Mail, Android/iOS, Webmail).
    Eigenschaften, Speicherbegrenzungen, Spam-Filter, E-Mail-Weiterleitung, Pflichten zur Nutzung der studentischen E-Mail im universitären Kontext.
    Hilfen bei überfülltem Postfach, Sicherheitsaspekte.
    eCampus Portal
    Funktionsübersicht des eCampus (zentrales Studierendenportal der Uni Göttingen): Zugang zu UniVZ, FlexNow, Stud.IP, persönlichen Kalender, Newsfeeds, Portlets etc.
    FAQs, Selfservice-Funktionen (SB-Funktionen), elektronische Formulare (z.B. Prüfungsrücktritt, Fristverlängerung bei Krankheit, Anerkennung von Prüfungsleistungen).
    Verwaltung und Personalisierung der Startseite (Portlets, Registerkarten, Layouts).
    Technikhilfen bei Loginproblemen oder Fehlern.
    Stud.IP & HISinOne EXA
    Nutzung und Zugang zum Lernmanagementsystem Stud.IP und zum Campusmanagementsystem HISinOne EXA (Vorlesungsverzeichnis, Personen-, Raum- und Veranstaltungsverwaltung).
    Zusammenhänge, Datenübertragung und Zusammenspiel der Systeme (z.B. UniVZ → Stud.IP).
    Studienausweis/Chipkarte
    Funktionen und Ausgabe des Studienausweises („Chipkarte“): Identifikation, Zahlfunktion, Semesterticket, Bibliotheksausweis, Kultursemesterticket, Heimvorteil.
    Vorgehen bei Verlust, Beschädigung oder Neuausstellung, Sperrung, Verlängerung.
    Besonderheiten bei der Rückmeldung, Semesterbeitrag, Immatrikulation, Beurlaubung.
    Drucken, Kopieren, Scannen
    Standorte und Nutzungsmöglichkeiten der studIT-PCs, Multifunktionsgeräte, Drucker (Follow-Me Printing, Direktdruck etc.), Kopierer und Scanner.
    Aufladen und Verwalten des Druckguthabens, Preise, Rückerstattungen, Freidruckguthaben, Übertragung von Guthaben.
    Nutzung verschiedener Endgeräte (Windows, MacOS, Linux, Android, iOS, USB-Stick, E-Mail-to-Print).
    Hinweise zu Posterdruck, Sondersoftware (z.B. für Blinde, Videobearbeitung).
    Netzwerke & Internet
    WLAN-Nutzung auf dem Campus (eduroam, GuestOnCampus, Einwahl für unterschiedliche Betriebssysteme).
    VPN-Nutzung (inkl. eduVPN, Anleitung und Sicherheitsaspekte, Mehrfaktor-Authentifizierung).
    Hinweise zu Verschlüsselung und Netzwerksicherheit.
    Datenspeicherung & -zugriff
    Informationen zu Homeverzeichnis (persönlicher Speicherplatz), Cloudspeicher (ownCloud), Netzlaufwerkverbindungen mit verschiedenen Betriebssystemen.
    Datensicherung/Backup, technische Anleitungen für den remote Zugriff (z.B. mit WinSCP, Cyberduck).
    Campussoftware & Zusatzdienste
    Verfügbare Software (Campuslizenzen: Office 365, Citavi, CorelDRAW, Statistik-Programme wie SAS/SPSS/Statistica, ATLAS.ti, MindManager).
    Installationsanleitungen und Support, Zugangsbedingungen, Besonderheiten bei Nutzung auf studIT-PCs und privat.
    Links und Beschreibungen zu Angeboten für Forschung und Lehre.
    Raum- und Schließfachreservierung (LSG/SUB)
    Anleitung und Bedingungen für die Reservierung von Arbeitsräumen und Schließfächern über eCampus/mobile Apps.
    Punktesystem, Einladung von Kommiliton:innen, Verwaltung, Stornierung.
    Bibliotheksdienste/SUB
    PCs und Internetzugang in der SUB/Bibliotheken, Nutzung als Studierende:r und Gast.
    Nutzung von Buchscannern, Spezialscannern, Kopier- und Druckkonten für Privatnutzer:innen.
    Kurse & Weiterbildung
    Angebot und Anmeldung zu ZESS-IT-Kursen für den Erwerb von Computerkenntnissen (z.B. Office, Grafikprogramme, HTML/CSS).
    Hinweise zur Anerkennung im Studium, Prüfungen und Credit Points.
    UniBlogs
    Informationen zu den Uni Göttingen Blogs für Studierende, Alumni, Lehrende und Organisationen: Anmeldung, Gestaltung, Nutzung von Plugins und Themes, Medienverwaltung.
    Glossar und Begriffserklärungen
    Bedeutungs- und Begriffserklärungen zu IT-relevanten und universitären Begriffen (z.B. LRC, FlexNow, Chipkarte, Homeverzeichnis, eduroam, VPN).
    Datenschutz & rechtliche Hinweise
    Datenschutzerklärungen gemäß DSGVO, Pflichtangaben, Ablauffristen für Speicherungen und Rechte betroffener Personen.
    Links und Hinweise zu offiziellen Ordnungen, Benutzungsordnungen.
    • Wenn eine Anleitung mehrere Varianten für unterschiedliche Betriebssysteme enthält und das Betriebssystem nicht genannt ist, MUSST du zuerst das Tool `ask` mit einer Rückfrage nach dem Betriebssystem aufrufen, bevor du das Tool `retrieve` nutzt.
    • Verwende `retrieve` erst, nachdem du die notwendige Spezifizierung erhalten hast (z. B. Betriebssystem, Gerätetyp).
    • Wenn du eine fachliche Frage beantwortest, RUFE IMMER zuerst das Tool `retrieve` auf und beantworte erst dann. Antworte niemals mit Fakten aus anderen Quellen.
    • Falls keine relevante Information gefunden wird, verweise auf offizielle Kontakte (z.B. Link oder E‑Mail).
    • Alle Fragen beziehen sich auf die Georg-August-Universität Göttingen.
    HALLUZINIERE NICHT.\n""")

# ----------------------------------------------------------------------
# 3.   Vektor‑Datenbank + LLM‑Instanz
# ----------------------------------------------------------------------

embedder = AcademicCloudEmbeddings(
    api_key=st.secrets["GWDG_API_KEY"],
    url=st.secrets["BASE_URL_EMBEDDINGS"],
)

vector_store = FAISS.load_local(
    str(VECTORSTORE_DIR),
    embedder,
    allow_dangerous_deserialization=True,
)

# load parent doc-store
with open(PARENT_STORE_PATH, "rb") as f:
    parent_store = pickle.load(f)

# re-create the *same* splitter used at index time
splitter = RecursiveCharacterTextSplitter(
    chunk_size    = 1000,
    chunk_overlap = 200,
    separators    = ["\n\n", "\n", " ", ""],
)
# retriever to use everywhere
retriever = ParentDocumentRetriever(
    vectorstore    = vector_store,
    docstore       = parent_store,
    child_splitter = splitter,
    search_kwargs  = {"k": 4},
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
    context: List[Document] # für spätere Anzeige der Quellen
    asked: bool = False  # ob eine Rückfrage gestellt wurde
    current_tool_docs: List[Document] = [] # für spätere Anzeige der Quellen

@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Sucht passende Wiki‑Dokumente per FAISS und liefert sie zurück."""
    docs = retriever.get_relevant_documents(query)
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
            AIMessage(content=f"""
                        Antworte **ausschließlich basierend auf den folgenden Ausschnitten**.
                        Nutze möglichst wörtliche Formulierungen und gib **keine Informationen wieder**, die **nicht explizit** in den Abschnitten genannt werden. 
                        **Erfinde keine zusätzlichen Informationen.**
                        Hier sind die kontextuellen Abschnitte:

                        {ctx_blocks}
                        """)
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
