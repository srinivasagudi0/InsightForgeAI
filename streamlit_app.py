from __future__ import annotations

import hashlib
import math
import os
import re
from html import escape
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from document_loader import (
    DocumentLoadError,
    LoadedDocument,
    load_document,
    load_text_input,
    load_uploaded_document,
)
from intel import DocumentSession, create_document_session
from report_exporters import (
    ReportSection,
    build_report_docx,
    build_report_markdown,
    build_report_pdf,
)


APP_TITLE = "InsightForge"
TASK_ORDER = [
    "summary",
    "key_information",
    "research_brief",
    "action_items",
    "feedback",
]
TASK_TITLES = {
    "summary": "Executive Summary",
    "key_information": "Key Highlights",
    "research_brief": "Research Brief",
    "action_items": "Next Moves",
    "feedback": "Editorial Review",
}
GRAPH_COLORS = [
    "#1d3557",
    "#457b9d",
    "#6d9773",
    "#c2843f",
    "#9f4f3f",
    "#7b5c89",
]


def resolve_backend_api_key(
    secrets_source: Any | None = None,
    env: Any | None = None,
) -> str | None:
    secrets_candidate = st.secrets if secrets_source is None else secrets_source
    env_candidate = os.environ if env is None else env

    for source in (secrets_candidate, env_candidate):
        value = _read_mapping_value(source, "OPENAI_API_KEY")
        if value:
            return value
    return None


def _read_mapping_value(source: Any, key: str) -> str | None:
    if source is None:
        return None

    getter = getattr(source, "get", None)
    if callable(getter):
        try:
            value = getter(key)
        except Exception:
            value = None
        normalized = _normalize_text(value)
        if normalized:
            return normalized

    try:
        value = source[key]
    except Exception:
        return None
    return _normalize_text(value)


def _normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def initialize_state() -> None:
    defaults = {
        "loaded_document": None,
        "document_signature": None,
        "document_session": None,
        "document_session_signature": None,
        "session_error": None,
        "outputs": {},
        "graph_data": None,
        "analysis_focus": "",
        "report_title": "InsightForge Brief",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@500;600&family=Manrope:wght@400;500;600;700&display=swap');

        :root {
            --canvas: #f5efe6;
            --canvas-soft: #fbf8f3;
            --ink: #18212b;
            --muted: #5f6a75;
            --line: rgba(24, 33, 43, 0.10);
            --accent: #1d3557;
            --accent-soft: rgba(29, 53, 87, 0.09);
            --accent-warm: #c2843f;
            --card: rgba(255, 255, 255, 0.72);
            --shadow: 0 18px 48px rgba(24, 33, 43, 0.08);
            --radius-xl: 28px;
            --radius-lg: 20px;
            --radius-md: 14px;
        }

        html, body, [class*="css"] {
            font-family: "Manrope", "Avenir Next", "Segoe UI", sans-serif;
            color: var(--ink);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(194, 132, 63, 0.18), transparent 32%),
                radial-gradient(circle at top right, rgba(69, 123, 157, 0.14), transparent 28%),
                linear-gradient(180deg, #f8f1e7 0%, #f4efe8 45%, #f7f4ef 100%);
        }

        .block-container {
            max-width: 1180px;
            padding-top: 2.1rem;
            padding-bottom: 3.5rem;
        }

        header[data-testid="stHeader"] {
            background: transparent;
            height: 0;
        }

        #MainMenu, footer {
            visibility: hidden;
        }

        .hero-card {
            background: linear-gradient(145deg, rgba(255, 255, 255, 0.88), rgba(255, 255, 255, 0.62));
            border: 1px solid rgba(255, 255, 255, 0.85);
            border-radius: var(--radius-xl);
            box-shadow: var(--shadow);
            padding: 2rem 2rem 1.7rem;
            position: relative;
            overflow: hidden;
        }

        .hero-card::after {
            content: "";
            position: absolute;
            inset: auto -60px -80px auto;
            width: 220px;
            height: 220px;
            border-radius: 999px;
            background: radial-gradient(circle, rgba(194, 132, 63, 0.18), transparent 68%);
            pointer-events: none;
        }

        .eyebrow {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.45rem 0.8rem;
            border-radius: 999px;
            background: rgba(24, 33, 43, 0.06);
            color: var(--ink);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 1rem;
        }

        .hero-title {
            font-family: "Fraunces", "Iowan Old Style", Georgia, serif;
            font-size: clamp(2.5rem, 4.7vw, 4.3rem);
            line-height: 0.95;
            margin: 0;
            letter-spacing: -0.04em;
        }

        .hero-copy {
            color: var(--muted);
            font-size: 1.05rem;
            line-height: 1.7;
            max-width: 48rem;
            margin: 1rem 0 0;
        }

        .panel-card {
            background: var(--card);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.85);
            border-radius: var(--radius-xl);
            box-shadow: var(--shadow);
            padding: 1.4rem;
        }

        .stack-card {
            background: linear-gradient(160deg, rgba(29, 53, 87, 0.92), rgba(49, 83, 124, 0.85));
            color: white;
            border-radius: var(--radius-xl);
            box-shadow: var(--shadow);
            padding: 1.5rem;
        }

        .stack-card h3,
        .panel-card h3,
        .result-heading {
            margin: 0;
            font-size: 1.05rem;
            font-weight: 700;
        }

        .stack-card p,
        .panel-card p,
        .muted-copy {
            margin: 0.35rem 0 0;
            line-height: 1.6;
            color: inherit;
        }

        .stack-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
            margin-top: 1rem;
        }

        .stack-pill {
            background: rgba(255, 255, 255, 0.10);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: var(--radius-lg);
            padding: 0.9rem;
        }

        .stack-pill strong {
            display: block;
            font-size: 0.95rem;
            margin-bottom: 0.2rem;
        }

        .stack-pill span {
            display: block;
            color: rgba(255, 255, 255, 0.76);
            font-size: 0.85rem;
            line-height: 1.45;
        }

        .section-title {
            font-family: "Fraunces", "Iowan Old Style", Georgia, serif;
            font-size: 1.85rem;
            letter-spacing: -0.03em;
            margin-bottom: 0.25rem;
        }

        .section-copy {
            color: var(--muted);
            margin-bottom: 1rem;
        }

        .preview-shell {
            margin-top: 1rem;
            padding: 1rem;
            border-radius: var(--radius-lg);
            background: rgba(24, 33, 43, 0.03);
            border: 1px solid var(--line);
        }

        .preview-shell pre {
            margin: 0;
            white-space: pre-wrap;
            font-family: "SFMono-Regular", "Menlo", monospace;
            font-size: 0.88rem;
            line-height: 1.55;
            color: var(--ink);
        }

        .feature-list {
            display: grid;
            gap: 0.75rem;
            margin-top: 1rem;
        }

        .feature-item {
            padding: 0.95rem 1rem;
            border-radius: var(--radius-lg);
            background: rgba(24, 33, 43, 0.03);
            border: 1px solid var(--line);
        }

        .feature-item strong {
            display: block;
            margin-bottom: 0.15rem;
        }

        .feature-item span {
            display: block;
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.55;
        }

        .stRadio > div {
            gap: 0.6rem;
        }

        .stRadio label {
            border-radius: 999px;
        }

        .stTextInput input,
        .stTextArea textarea,
        .stFileUploader,
        .stSelectbox [data-baseweb="select"] > div,
        .stDownloadButton button,
        .stButton button {
            border-radius: 14px;
        }

        .stTextInput input,
        .stTextArea textarea,
        .stFileUploader > div,
        .stSelectbox [data-baseweb="select"] > div {
            background: rgba(255, 255, 255, 0.62);
            border: 1px solid rgba(24, 33, 43, 0.12);
        }

        .stButton button,
        .stDownloadButton button {
            border: 1px solid rgba(24, 33, 43, 0.10);
            background: rgba(255, 255, 255, 0.70);
            min-height: 2.9rem;
            font-weight: 700;
            box-shadow: 0 8px 22px rgba(24, 33, 43, 0.06);
        }

        .stButton button[kind="primary"],
        .stFormSubmitButton button[kind="primary"] {
            background: linear-gradient(135deg, #1d3557, #2c4f76);
            color: white;
            border: none;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
            margin-bottom: 1rem;
        }

        .stTabs [data-baseweb="tab"] {
            height: 2.8rem;
            background: rgba(255, 255, 255, 0.56);
            border-radius: 999px;
            padding: 0 1rem;
            border: 1px solid rgba(24, 33, 43, 0.08);
        }

        .stTabs [aria-selected="true"] {
            background: rgba(29, 53, 87, 0.96);
            color: white;
        }

        .result-card {
            background: rgba(255, 255, 255, 0.66);
            border: 1px solid rgba(24, 33, 43, 0.08);
            border-radius: var(--radius-lg);
            padding: 1.1rem 1.2rem 0.8rem;
            margin-bottom: 1rem;
        }

        .result-kicker {
            color: var(--muted);
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.55rem;
        }

        .chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 0.85rem;
        }

        .chip {
            display: inline-flex;
            align-items: center;
            padding: 0.45rem 0.75rem;
            border-radius: 999px;
            background: rgba(24, 33, 43, 0.05);
            border: 1px solid rgba(24, 33, 43, 0.08);
            font-size: 0.82rem;
            font-weight: 700;
            color: var(--ink);
        }

        .placeholder-card {
            border-radius: var(--radius-xl);
            padding: 1.5rem;
            border: 1px dashed rgba(24, 33, 43, 0.16);
            background: rgba(255, 255, 255, 0.42);
            color: var(--muted);
        }

        @media (max-width: 960px) {
            .stack-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    initialize_state()
    inject_styles()
    render_page()


def render_page() -> None:
    render_hero()

    intake_column, document_column = st.columns([1.15, 0.85], gap="large")
    with intake_column:
        render_intake_panel()
    with document_column:
        render_document_panel()

    if st.session_state.loaded_document is not None:
        render_workspace()
    else:
        render_empty_workspace()


def render_hero() -> None:
    left_column, right_column = st.columns([1.35, 0.95], gap="large")

    with left_column:
        st.markdown(
            """
            <section class="hero-card">
                <div class="eyebrow">Document Workspace</div>
                <h1 class="hero-title">Readable answers from dense documents.</h1>
                <p class="hero-copy">
                    Upload a paper, memo, report, or draft. Move from raw text to crisp takeaways,
                    a conversation-ready view, and clean deliverables without the clutter of a demo tool.
                </p>
            </section>
            """,
            unsafe_allow_html=True,
        )

    with right_column:
        st.markdown(
            """
            <section class="stack-card">
                <h3>Built for the way people actually review material</h3>
                <p>Keep the screen calm, keep the language sharp, and make the useful parts easy to find.</p>
                <div class="stack-grid">
                    <div class="stack-pill">
                        <strong>Quick brief</strong>
                        <span>Get to the point fast when time is tight.</span>
                    </div>
                    <div class="stack-pill">
                        <strong>Live dialogue</strong>
                        <span>Ask precise follow-ups without losing the thread.</span>
                    </div>
                    <div class="stack-pill">
                        <strong>Presentation-ready</strong>
                        <span>Package outputs into a clean report in one place.</span>
                    </div>
                </div>
            </section>
            """,
            unsafe_allow_html=True,
        )


def render_intake_panel() -> None:
    st.markdown(
        """
        <div class="panel-card">
            <div class="section-title">Bring in a document</div>
            <div class="section-copy">Choose the fastest way to open your workspace.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    source_mode = st.radio(
        "Source",
        options=["Upload file", "Paste text", "Use local path"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if source_mode == "Upload file":
        render_upload_form()
    elif source_mode == "Paste text":
        render_paste_form()
    else:
        render_path_form()


def render_upload_form() -> None:
    with st.form("upload_form", clear_on_submit=False):
        uploaded_file = st.file_uploader(
            "Upload a PDF or text file",
            type=["pdf", "txt", "md", "rst", "csv", "json"],
            help="PDF and text-based formats work best.",
        )
        submitted = st.form_submit_button(
            "Open workspace",
            type="primary",
            use_container_width=True,
        )

    if not submitted:
        return
    if uploaded_file is None:
        st.warning("Choose a file first.")
        return

    try:
        document = load_uploaded_document(uploaded_file.name, uploaded_file.getvalue())
    except DocumentLoadError as exc:
        st.error(str(exc))
        return

    set_loaded_document(document)
    st.rerun()


def render_paste_form() -> None:
    with st.form("paste_form", clear_on_submit=False):
        document_name = st.text_input(
            "Document name",
            value="Working Notes",
            placeholder="Quarterly planning memo",
        )
        text = st.text_area(
            "Paste content",
            height=250,
            placeholder="Paste the document body here...",
        )
        submitted = st.form_submit_button(
            "Open workspace",
            type="primary",
            use_container_width=True,
        )

    if not submitted:
        return
    if not text.strip():
        st.warning("Paste some content to continue.")
        return

    try:
        document = load_text_input(document_name.strip() or "Untitled Document", text)
    except DocumentLoadError as exc:
        st.error(str(exc))
        return

    set_loaded_document(document)
    st.rerun()


def render_path_form() -> None:
    with st.form("path_form", clear_on_submit=False):
        file_path = st.text_input(
            "Local file path",
            placeholder="~/Documents/research-brief.pdf",
        )
        submitted = st.form_submit_button(
            "Open workspace",
            type="primary",
            use_container_width=True,
        )

    if not submitted:
        return
    if not file_path.strip():
        st.warning("Enter a file path to continue.")
        return

    try:
        document = load_document(file_path)
    except DocumentLoadError as exc:
        st.error(str(exc))
        return

    set_loaded_document(document)
    st.rerun()


def render_document_panel() -> None:
    document = st.session_state.loaded_document
    if document is None:
        st.markdown(
            """
            <section class="panel-card">
                <div class="section-title">What the workspace gives you</div>
                <div class="feature-list">
                    <div class="feature-item">
                        <strong>Faster reading</strong>
                        <span>Pull the main argument, evidence, and implications into a view you can scan quickly.</span>
                    </div>
                    <div class="feature-item">
                        <strong>Better follow-up questions</strong>
                        <span>Switch from passive reading to targeted conversation once the document is open.</span>
                    </div>
                    <div class="feature-item">
                        <strong>Cleaner handoff</strong>
                        <span>Turn working analysis into a portable report without rebuilding everything from scratch.</span>
                    </div>
                </div>
            </section>
            """,
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f"""
        <section class="panel-card">
            <div class="eyebrow">Current document</div>
            <div class="section-title">{escape(document.name)}</div>
            <p class="section-copy">{escape(document.source_type)} source prepared for exploration.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    metric_columns = st.columns(3)
    metric_columns[0].metric("Words", f"{document.word_count:,}")
    if document.page_count is not None:
        metric_columns[1].metric("Pages", f"{document.page_count:,}")
    else:
        metric_columns[1].metric("Characters", f"{document.char_count:,}")
    metric_columns[2].metric("Source", document.source_type)

    if document.used_ocr:
        st.caption("Image-based pages were read and included in the workspace.")

    st.markdown(
        f"""
        <div class="preview-shell">
            <pre>{escape(_preview_text(document.content))}</pre>
        </div>
        """,
        unsafe_allow_html=True,
    )

    action_columns = st.columns(2)
    if action_columns[0].button("Clear workspace", use_container_width=True):
        clear_workspace()
        st.rerun()
    if action_columns[1].button("Reset conversation", use_container_width=True):
        reset_conversation()
        st.rerun()


def render_empty_workspace() -> None:
    st.write("")
    st.markdown(
        """
        <section class="placeholder-card">
            <div class="result-kicker">Ready when you are</div>
            Open a document to unlock briefs, key highlights, a relationship map, live Q&A, and exportable reports.
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_workspace() -> None:
    document = st.session_state.loaded_document
    if document is None:
        return

    st.write("")
    st.markdown(
        """
        <div class="panel-card">
            <div class="section-title">Workspace</div>
            <div class="section-copy">Pick a lens, generate what you need, and keep the useful output in one place.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.text_input(
        "Lens",
        key="analysis_focus",
        placeholder="Optional: risks, roadmap impact, methodology, findings...",
        help="Use a lens to make the outputs more specific.",
    )

    if st.session_state.session_error:
        st.warning(st.session_state.session_error)

    tab_overview, tab_chat, tab_map, tab_export = st.tabs(
        ["Overview", "Chat", "Map", "Export"]
    )

    with tab_overview:
        render_overview_tab()
    with tab_chat:
        render_chat_tab()
    with tab_map:
        render_map_tab()
    with tab_export:
        render_export_tab()


def render_overview_tab() -> None:
    focus = st.session_state.analysis_focus.strip()

    row_one = st.columns(3)
    row_two = st.columns(3)

    if row_one[0].button("Executive Summary", use_container_width=True):
        run_text_task("summary", focus)
    if row_one[1].button("Key Highlights", use_container_width=True):
        run_text_task("key_information", focus)
    if row_one[2].button("Research Brief", use_container_width=True):
        run_text_task("research_brief", focus)
    if row_two[0].button("Next Moves", use_container_width=True):
        run_text_task("action_items", focus)
    if row_two[1].button("Editorial Review", use_container_width=True):
        run_text_task("feedback", focus)
    if row_two[2].button("Relationship Map", use_container_width=True):
        run_graph_task(focus)

    outputs = st.session_state.outputs
    graph_data = st.session_state.graph_data

    if not outputs and graph_data is None:
        st.write("")
        st.markdown(
            """
            <section class="placeholder-card">
                <div class="result-kicker">Start with a first pass</div>
                Use the buttons above to create a summary, pull the most important facts, or shape the document into a brief.
            </section>
            """,
            unsafe_allow_html=True,
        )
        return

    for task_key in TASK_ORDER:
        content = outputs.get(task_key)
        if content:
            render_result_card(TASK_TITLES[task_key], content)

    if graph_data is not None:
        render_result_card("Relationship Map", graph_data.get("summary", ""))


def render_chat_tab() -> None:
    session = ensure_document_session()
    if session is None:
        st.markdown(
            """
            <section class="placeholder-card">
                <div class="result-kicker">Conversation unavailable</div>
                This workspace cannot start a document conversation right now.
            </section>
            """,
            unsafe_allow_html=True,
        )
        return

    if session.has_restored_memory():
        st.caption("Previous conversation context was restored for this document.")

    if not session.history:
        st.markdown(
            """
            <section class="placeholder-card">
                <div class="result-kicker">Ask the first question</div>
                Try a targeted prompt such as “What changed?”, “Where are the risks?”, or “What should a product team care about?”.
            </section>
            """,
            unsafe_allow_html=True,
        )

    for message in session.history:
        role = "user" if message.get("role") == "user" else "assistant"
        with st.chat_message(role):
            st.markdown(message.get("content", ""))

    prompt = st.chat_input("Ask about the document")
    if not prompt:
        return

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking through the document..."):
            try:
                answer = session.ask(prompt)
            except RuntimeError:
                st.markdown("That request could not be completed right now.")
                return
            st.markdown(answer)

    st.rerun()


def render_map_tab() -> None:
    focus = st.session_state.analysis_focus.strip()
    summary_column, action_column = st.columns([1.15, 0.85], gap="large")

    with summary_column:
        graph_data = st.session_state.graph_data
        if graph_data is None:
            st.markdown(
                """
                <section class="placeholder-card">
                    <div class="result-kicker">Map the document</div>
                    Create a relationship map to surface the main entities, ideas, methods, and outcomes in one view.
                </section>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.plotly_chart(
                build_graph_figure(graph_data),
                use_container_width=True,
                config={"displayModeBar": False},
            )

    with action_column:
        st.markdown(
            """
            <section class="panel-card">
                <div class="section-title">Relationship map</div>
                <p class="section-copy">A visual pass helps when the document has many moving parts or repeated actors.</p>
            </section>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Generate map", type="primary", use_container_width=True):
            run_graph_task(focus)

        graph_data = st.session_state.graph_data
        if graph_data is not None:
            st.markdown(
                f"""
                <section class="result-card">
                    <div class="result-kicker">Map summary</div>
                    <div class="result-heading">{escape(graph_data.get("title", "Relationship Map"))}</div>
                    <p class="muted-copy">{escape(graph_data.get("summary", ""))}</p>
                </section>
                """,
                unsafe_allow_html=True,
            )
            render_chip_row(
                [
                    f"{len(graph_data.get('nodes', []))} nodes",
                    f"{len(graph_data.get('edges', []))} links",
                ]
            )

    graph_data = st.session_state.graph_data
    if graph_data is None:
        return

    node_frame = pd.DataFrame(graph_data.get("nodes", []))
    edge_frame = pd.DataFrame(graph_data.get("edges", []))
    table_one, table_two = st.columns(2, gap="large")
    with table_one:
        st.markdown("**Nodes**")
        st.dataframe(node_frame, use_container_width=True, hide_index=True)
    with table_two:
        st.markdown("**Edges**")
        st.dataframe(edge_frame, use_container_width=True, hide_index=True)


def render_export_tab() -> None:
    document = st.session_state.loaded_document
    if document is None:
        return

    sections = build_report_sections()
    if not sections:
        st.markdown(
            """
            <section class="placeholder-card">
                <div class="result-kicker">Nothing to export yet</div>
                Generate at least one output first, then come back here for a clean report package.
            </section>
            """,
            unsafe_allow_html=True,
        )
        return

    title_column, chip_column = st.columns([1.2, 0.8], gap="large")
    with title_column:
        st.text_input(
            "Report title",
            key="report_title",
            placeholder="InsightForge Brief",
        )
    with chip_column:
        render_chip_row([section.title for section in sections])

    report_title = st.session_state.report_title.strip() or "InsightForge Brief"
    markdown_report = build_report_markdown(report_title, document.name, sections)
    docx_report = build_report_docx(report_title, document.name, sections)
    pdf_report = build_report_pdf(report_title, document.name, sections)
    base_name = _safe_filename(report_title)

    button_columns = st.columns(3)
    button_columns[0].download_button(
        "Download Markdown",
        markdown_report.encode("utf-8"),
        file_name=f"{base_name}.md",
        mime="text/markdown",
        use_container_width=True,
    )
    button_columns[1].download_button(
        "Download DOCX",
        docx_report,
        file_name=f"{base_name}.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        use_container_width=True,
    )
    button_columns[2].download_button(
        "Download PDF",
        pdf_report,
        file_name=f"{base_name}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

    with st.expander("Preview report", expanded=True):
        st.markdown(markdown_report)


def render_result_card(title: str, content: str) -> None:
    st.markdown(
        f"""
        <section class="result-card">
            <div class="result-kicker">Generated view</div>
            <div class="result-heading">{escape(title)}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(content)


def render_chip_row(items: list[str]) -> None:
    if not items:
        return
    chips = "".join(f"<span class=\"chip\">{escape(item)}</span>" for item in items)
    st.markdown(f"<div class=\"chip-row\">{chips}</div>", unsafe_allow_html=True)


def set_loaded_document(document: LoadedDocument) -> None:
    signature = build_document_signature(document)
    current_signature = st.session_state.document_signature

    st.session_state.loaded_document = document
    st.session_state.document_signature = signature
    st.session_state.report_title = f"{_document_stem(document.name)} Brief"

    if signature == current_signature:
        return

    st.session_state.document_session = None
    st.session_state.document_session_signature = None
    st.session_state.session_error = None
    st.session_state.outputs = {}
    st.session_state.graph_data = None
    st.session_state.analysis_focus = ""


def clear_workspace() -> None:
    st.session_state.loaded_document = None
    st.session_state.document_signature = None
    st.session_state.document_session = None
    st.session_state.document_session_signature = None
    st.session_state.session_error = None
    st.session_state.outputs = {}
    st.session_state.graph_data = None
    st.session_state.analysis_focus = ""
    st.session_state.report_title = "InsightForge Brief"


def reset_conversation() -> None:
    session = st.session_state.document_session
    if session is not None:
        session.clear_memory()


def ensure_document_session() -> DocumentSession | None:
    document = st.session_state.loaded_document
    if document is None:
        return None

    cached_session = st.session_state.document_session
    signature = build_document_signature(document)
    if (
        cached_session is not None
        and st.session_state.document_session_signature == signature
    ):
        return cached_session

    api_key = resolve_backend_api_key()
    if not api_key:
        st.session_state.session_error = "Document tools are unavailable right now."
        return None

    try:
        session = create_document_session(
            document=document.content,
            document_name=document.name,
            api_key=api_key,
        )
    except RuntimeError:
        st.session_state.session_error = "That request could not be completed right now."
        return None

    st.session_state.document_session = session
    st.session_state.document_session_signature = signature
    st.session_state.session_error = None
    return session


def run_text_task(task_key: str, focus: str) -> None:
    session = ensure_document_session()
    if session is None:
        return

    handlers = {
        "summary": session.generate_summary,
        "key_information": session.extract_key_information,
        "research_brief": session.generate_research_brief,
        "action_items": session.extract_action_items,
        "feedback": session.provide_feedback,
    }

    action = handlers[task_key]
    with st.spinner(f"Building {TASK_TITLES[task_key].lower()}..."):
        try:
            result = action(focus)
        except RuntimeError:
            st.session_state.session_error = "That request could not be completed right now."
            return

    outputs = dict(st.session_state.outputs)
    outputs[task_key] = result
    st.session_state.outputs = outputs


def run_graph_task(focus: str) -> None:
    session = ensure_document_session()
    if session is None:
        return

    with st.spinner("Building relationship map..."):
        try:
            graph_data = session.build_graph_data(focus)
        except RuntimeError:
            st.session_state.session_error = "That request could not be completed right now."
            return

    st.session_state.graph_data = graph_data


def build_document_signature(document: LoadedDocument) -> str:
    digest = hashlib.sha1(
        f"{document.name}\n{document.word_count}\n{document.content}".encode("utf-8")
    ).hexdigest()
    return digest[:24]


def _preview_text(text: str, limit: int = 1200) -> str:
    normalized = text.strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit].rstrip() + "\n\n..."


def build_report_sections() -> list[ReportSection]:
    sections: list[ReportSection] = []
    outputs = st.session_state.outputs

    for task_key in TASK_ORDER:
        content = outputs.get(task_key)
        if content:
            sections.append(ReportSection(TASK_TITLES[task_key], content))

    graph_data = st.session_state.graph_data
    if graph_data is not None:
        sections.append(
            ReportSection(
                "Relationship Map",
                build_graph_report_text(graph_data),
            )
        )
    return sections


def build_graph_report_text(graph_data: dict[str, Any]) -> str:
    label_map = {
        str(node.get("id")): str(node.get("label"))
        for node in graph_data.get("nodes", [])
        if isinstance(node, dict)
    }
    lines = [graph_data.get("summary", "").strip()]
    if graph_data.get("nodes"):
        lines.append("")
        lines.append("Nodes")
        for node in graph_data["nodes"]:
            lines.append(f"- {node['label']} ({node['group']})")
    if graph_data.get("edges"):
        lines.append("")
        lines.append("Edges")
        for edge in graph_data["edges"]:
            source = label_map.get(edge["source"], edge["source"])
            target = label_map.get(edge["target"], edge["target"])
            lines.append(f"- {source} -> {target}: {edge['label']}")
    return "\n".join(line for line in lines if line is not None).strip()


def build_graph_figure(graph_data: dict[str, Any]) -> go.Figure:
    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])
    if not nodes:
        return go.Figure()

    position_map = build_node_positions(nodes)
    color_map = build_group_color_map(nodes)

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for edge in edges:
        start = position_map.get(edge["source"])
        end = position_map.get(edge["target"])
        if start is None or end is None:
            continue
        edge_x.extend([start[0], end[0], None])
        edge_y.extend([start[1], end[1], None])

    node_x = [position_map[node["id"]][0] for node in nodes]
    node_y = [position_map[node["id"]][1] for node in nodes]
    node_text = [node["label"] for node in nodes]
    node_groups = [node["group"] for node in nodes]
    node_colors = [color_map[node["group"]] for node in nodes]

    figure = go.Figure()
    if edge_x:
        figure.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line={"width": 1.8, "color": "rgba(29, 53, 87, 0.22)"},
                hoverinfo="skip",
                showlegend=False,
            )
        )

    figure.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="top center",
            hovertemplate="<b>%{text}</b><br>%{customdata}<extra></extra>",
            customdata=node_groups,
            marker={
                "size": 26,
                "color": node_colors,
                "line": {"width": 2, "color": "#ffffff"},
            },
            showlegend=False,
        )
    )

    figure.update_layout(
        margin={"l": 12, "r": 12, "t": 12, "b": 12},
        height=520,
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return figure


def build_node_positions(nodes: list[dict[str, str]]) -> dict[str, tuple[float, float]]:
    total = max(len(nodes), 1)
    positions: dict[str, tuple[float, float]] = {}
    for index, node in enumerate(nodes):
        angle = (2 * math.pi * index) / total
        radius = 1.0 + (0.16 if index % 2 else 0.0)
        positions[node["id"]] = (
            math.cos(angle) * radius,
            math.sin(angle) * radius,
        )
    return positions


def build_group_color_map(nodes: list[dict[str, str]]) -> dict[str, str]:
    groups = []
    for node in nodes:
        group = node.get("group", "concept")
        if group not in groups:
            groups.append(group)
    return {
        group: GRAPH_COLORS[index % len(GRAPH_COLORS)]
        for index, group in enumerate(groups)
    }


def _safe_filename(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return slug or "insightforge-brief"


def _document_stem(value: str) -> str:
    stem = value.rsplit(".", 1)[0].strip()
    return stem or "InsightForge"


if __name__ == "__main__":
    main()
