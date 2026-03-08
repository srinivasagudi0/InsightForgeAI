from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import OpenAI


DEFAULT_MODEL = "gpt-4o"
DEFAULT_RECENT_TURNS = 6
DEFAULT_MEMORY_NOTES = 20
DEFAULT_INLINE_WORD_LIMIT = 1800
DEFAULT_CHUNK_WORD_TARGET = 850
DEFAULT_CHUNK_OVERLAP_WORDS = 120
DEFAULT_MAX_CONTEXT_CHUNKS = 4
DEFAULT_MAX_GLOBAL_ANALYSIS_CHUNKS = 10
MEMORY_STORE = Path.home() / ".insightforge_ai" / "memory.json"

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
}


@dataclass(frozen=True)
class DocumentChunk:
    index: int
    label: str
    content: str
    word_count: int


@dataclass
class DocumentSession:
    document: str
    document_name: str = "document"
    api_key: str | None = None
    model: str = DEFAULT_MODEL
    max_recent_turns: int = DEFAULT_RECENT_TURNS
    max_memory_notes: int = DEFAULT_MEMORY_NOTES
    inline_word_limit: int = DEFAULT_INLINE_WORD_LIMIT
    chunk_word_target: int = DEFAULT_CHUNK_WORD_TARGET
    chunk_overlap_words: int = DEFAULT_CHUNK_OVERLAP_WORDS
    max_context_chunks: int = DEFAULT_MAX_CONTEXT_CHUNKS
    max_global_analysis_chunks: int = DEFAULT_MAX_GLOBAL_ANALYSIS_CHUNKS
    memory_store: Path = MEMORY_STORE
    client: OpenAI = field(init=False)
    history: list[dict[str, str]] = field(default_factory=list)
    memory_notes: list[str] = field(default_factory=list)
    analysis_cache: dict[str, list[str]] = field(default_factory=dict)
    restored_turns: int = field(init=False, default=0)
    restored_notes: int = field(init=False, default=0)
    session_id: str = field(init=False)
    chunks: list[DocumentChunk] = field(init=False, default_factory=list)
    document_word_count: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OpenAI API key is required.")

        self.client = OpenAI(api_key=api_key)
        self.document_word_count = len(self.document.split())
        self.chunks = self._split_document(self.document)
        self.session_id = self._build_session_id()
        self._load_memory()

    def ask(self, question: str) -> str:
        question = question.strip()
        if not question:
            return "Please enter a question."

        context = self._context_for_question(question)
        answer = self._request_text(
            user_prompt=question,
            task_instruction=(
                "Answer the user's question using the document context and the saved conversation memory. "
                "Treat follow-up questions as part of the same chat."
            ),
            document_context=context,
        )
        self._remember(question, answer)
        return answer

    def generate_summary(self, focus: str = "") -> str:
        prompt = "Generate a concise but complete summary of the document."
        if focus.strip():
            prompt += f" Focus on: {focus.strip()}."
        return self._run_document_text_task(
            task_key=f"summary::{focus.strip().lower()}",
            user_prompt=prompt,
            task_instruction=(
                "Write a clear summary grounded in the document. Prioritize factual coverage over generic phrasing."
            ),
            chunk_prompt=(
                "Summarize this section with the main ideas, evidence, methods, and notable limitations."
            ),
        )

    def extract_key_information(self, focus: str = "") -> str:
        prompt = (
            "Extract the key information from the document as a bullet list. Include important names, dates, facts, claims, and takeaways."
        )
        if focus.strip():
            prompt += f" Prioritize: {focus.strip()}."
        return self._run_document_text_task(
            task_key=f"key-information::{focus.strip().lower()}",
            user_prompt=prompt,
            task_instruction=(
                "Return structured key information that is easy to scan. Prefer bullets and short section headers."
            ),
            chunk_prompt=(
                "Extract the most important facts, actors, claims, methods, datasets, and outcomes from this section."
            ),
        )

    def generate_research_brief(self, focus: str = "") -> str:
        prompt = (
            "Create a research brief with the sections: Objective, Methods, Evidence, Results, Limitations, and Why It Matters."
        )
        if focus.strip():
            prompt += f" Focus on: {focus.strip()}."
        return self._run_document_text_task(
            task_key=f"research-brief::{focus.strip().lower()}",
            user_prompt=prompt,
            task_instruction=(
                "Write a crisp research brief that reads like analyst work, not generic summary text."
            ),
            chunk_prompt=(
                "Capture the objective, methods, evidence, findings, and limitations from this section."
            ),
        )

    def extract_action_items(self, focus: str = "") -> str:
        prompt = (
            "List the next actions, open questions, follow-up experiments, and implementation ideas implied by this document."
        )
        if focus.strip():
            prompt += f" Focus on: {focus.strip()}."
        return self._run_document_text_task(
            task_key=f"action-items::{focus.strip().lower()}",
            user_prompt=prompt,
            task_instruction=(
                "Turn the document into practical next steps. Group the output into clear action-oriented sections."
            ),
            chunk_prompt=(
                "Extract concrete next steps, open questions, implementation ideas, and risks from this section."
            ),
        )

    def provide_feedback(self, focus: str = "") -> str:
        prompt = (
            "Provide constructive feedback on the document's content, structure, clarity, and completeness."
        )
        if focus.strip():
            prompt += f" Focus especially on: {focus.strip()}."
        return self._run_document_text_task(
            task_key=f"feedback::{focus.strip().lower()}",
            user_prompt=prompt,
            task_instruction=(
                "Give practical feedback the author can act on. Separate strengths, issues, and suggested revisions."
            ),
            chunk_prompt=(
                "Review this section for clarity, structure, missing context, and writing quality."
            ),
        )

    def build_graph_data(self, focus: str = "") -> dict[str, Any]:
        prompt = (
            "Build a compact knowledge graph from the document. "
            "Return JSON only with keys: title, summary, nodes, edges. "
            "Each node must include id, label, group. "
            "Each edge must include source, target, label. "
            "Prefer 6 to 12 nodes and avoid duplicates."
        )
        if focus.strip():
            prompt += f" Emphasize relationships around: {focus.strip()}."

        payload = self._run_document_json_task(
            task_key=f"graph::{focus.strip().lower()}",
            user_prompt=prompt,
            task_instruction=(
                "Synthesize the document into graph-ready JSON that captures the most meaningful entities and relationships."
            ),
            chunk_prompt=(
                "Extract the important entities, methods, datasets, findings, and relationships from this section."
            ),
        )
        graph = self._normalize_graph_payload(payload)
        self._remember(
            prompt,
            f"Generated knowledge graph '{graph['title']}' with {len(graph['nodes'])} nodes and {len(graph['edges'])} edges.",
        )
        return graph

    def visualize_as_graph(self, focus: str = "") -> str:
        graph = self.build_graph_data(focus)
        label_map = {node["id"]: node["label"] for node in graph["nodes"]}
        lines = [f"## {graph['title']}", graph["summary"], "", "### Nodes"]
        for node in graph["nodes"]:
            lines.append(f"- {node['label']} ({node['group']})")
        lines.append("")
        lines.append("### Edges")
        for edge in graph["edges"]:
            source = label_map.get(edge["source"], edge["source"])
            target = label_map.get(edge["target"], edge["target"])
            lines.append(f"- {source} -> {target}: {edge['label']}")
        return "\n".join(lines).strip()

    def clear_memory(self) -> None:
        self.history.clear()
        self.memory_notes.clear()
        self.analysis_cache.clear()
        self.restored_turns = 0
        self.restored_notes = 0
        store = self._load_store()
        store.setdefault("sessions", {}).pop(self.session_id, None)
        self._save_store(store)

    def has_restored_memory(self) -> bool:
        return self.restored_turns > 0 or self.restored_notes > 0

    def _run_document_text_task(
        self,
        task_key: str,
        user_prompt: str,
        task_instruction: str,
        chunk_prompt: str,
    ) -> str:
        context = self._global_task_context(task_key, chunk_prompt)
        answer = self._request_text(
            user_prompt=user_prompt,
            task_instruction=task_instruction,
            document_context=context,
        )
        self._remember(user_prompt, answer)
        return answer

    def _run_document_json_task(
        self,
        task_key: str,
        user_prompt: str,
        task_instruction: str,
        chunk_prompt: str,
    ) -> dict[str, Any]:
        context = self._global_task_context(task_key, chunk_prompt)
        return self._request_json(
            user_prompt=user_prompt,
            task_instruction=task_instruction,
            document_context=context,
        )

    def _global_task_context(self, task_key: str, chunk_prompt: str) -> str:
        if not self._should_chunk():
            return self._full_document_context()

        if task_key in self.analysis_cache:
            notes = self.analysis_cache[task_key]
        else:
            notes = self._collect_chunk_notes(task_key, chunk_prompt)
            self.analysis_cache[task_key] = notes
        return "Section analyses derived from the document:\n\n" + "\n\n".join(notes)

    def _collect_chunk_notes(self, task_key: str, chunk_prompt: str) -> list[str]:
        notes: list[str] = []
        for chunk in self._grouped_chunks_for_global_tasks():
            note = self._request_text(
                user_prompt=chunk_prompt,
                task_instruction=(
                    "Analyze only the provided section. Return compact, content-dense bullets and avoid filler."
                ),
                document_context=f"{chunk.label}\n{chunk.content}",
                include_history=False,
                include_memory_notes=False,
            )
            notes.append(f"{chunk.label}\n{note}")
        return notes

    def _request_text(
        self,
        user_prompt: str,
        task_instruction: str,
        document_context: str,
        include_history: bool = True,
        include_memory_notes: bool = True,
    ) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self._build_messages(
                    user_prompt=user_prompt,
                    task_instruction=task_instruction,
                    document_context=document_context,
                    include_history=include_history,
                    include_memory_notes=include_memory_notes,
                ),
            )
        except Exception as exc:
            raise RuntimeError(f"OpenAI request failed: {exc}") from exc

        return (response.choices[0].message.content or "").strip()

    def _request_json(
        self,
        user_prompt: str,
        task_instruction: str,
        document_context: str,
    ) -> dict[str, Any]:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self._build_messages(
                    user_prompt=user_prompt,
                    task_instruction=task_instruction,
                    document_context=document_context,
                ),
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            raise RuntimeError(f"OpenAI request failed: {exc}") from exc

        raw_content = (response.choices[0].message.content or "").strip()
        try:
            payload = json.loads(raw_content)
        except json.JSONDecodeError as exc:
            raise RuntimeError("The model returned invalid JSON.") from exc

        if not isinstance(payload, dict):
            raise RuntimeError("The model did not return a JSON object.")
        return payload

    def _build_messages(
        self,
        user_prompt: str,
        task_instruction: str,
        document_context: str,
        include_history: bool = True,
        include_memory_notes: bool = True,
    ) -> list[dict[str, str]]:
        memory_block = (
            "\n".join(f"- {note}" for note in self.memory_notes)
            if include_memory_notes and self.memory_notes
            else "- No saved notes yet."
        )
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    f"You are InsightForge-AI, a document assistant for '{self.document_name}'.\n"
                    "Use the provided document context as the primary source of truth. "
                    "If the context does not support an answer, say that clearly.\n\n"
                    f"Task instruction: {task_instruction}\n\n"
                    f"Saved memory notes:\n{memory_block}\n\n"
                    f"Document context:\n{document_context}"
                ),
            }
        ]
        if include_history:
            messages.extend(self.history)
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def _context_for_question(self, question: str) -> str:
        if not self._should_chunk():
            return self._full_document_context()
        return self._format_chunks(self._select_relevant_chunks(question))

    def _full_document_context(self) -> str:
        return self.document

    def _should_chunk(self) -> bool:
        return (
            self.document_word_count > self.inline_word_limit
            and len(self.chunks) > 1
        )

    def _split_document(self, document: str) -> list[DocumentChunk]:
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", document) if part.strip()]
        if not paragraphs:
            paragraphs = [document.strip()]

        units = self._paragraph_units(paragraphs)
        chunks: list[DocumentChunk] = []
        current_units: list[str] = []
        current_words = 0

        for unit in units:
            unit_words = len(unit.split())
            if current_units and current_words + unit_words > self.chunk_word_target:
                chunks.append(self._make_chunk(len(chunks) + 1, current_units))
                overlap = self._tail_words(" ".join(current_units), self.chunk_overlap_words)
                current_units = [overlap, unit] if overlap else [unit]
                current_words = len(" ".join(current_units).split())
            else:
                current_units.append(unit)
                current_words += unit_words

        if current_units:
            chunks.append(self._make_chunk(len(chunks) + 1, current_units))

        return chunks or [
            DocumentChunk(
                index=1,
                label="Chunk 1",
                content=document.strip(),
                word_count=len(document.split()),
            )
        ]

    def _paragraph_units(self, paragraphs: list[str]) -> list[str]:
        units: list[str] = []
        for paragraph in paragraphs:
            words = paragraph.split()
            if len(words) <= self.chunk_word_target:
                units.append(paragraph)
                continue

            start = 0
            while start < len(words):
                end = min(len(words), start + self.chunk_word_target)
                units.append(" ".join(words[start:end]))
                if end == len(words):
                    break
                start = max(0, end - self.chunk_overlap_words)
        return units

    def _make_chunk(self, index: int, units: list[str]) -> DocumentChunk:
        content = "\n\n".join(part for part in units if part.strip()).strip()
        return DocumentChunk(
            index=index,
            label=f"Chunk {index}",
            content=content,
            word_count=len(content.split()),
        )

    def _tail_words(self, text: str, limit: int) -> str:
        words = text.split()
        if not words or limit <= 0:
            return ""
        return " ".join(words[-limit:])

    def _select_relevant_chunks(self, query: str) -> list[DocumentChunk]:
        query_tokens = self._keyword_tokens(query)
        if not query_tokens:
            return self.chunks[: self.max_context_chunks]

        scored: list[tuple[int, int, DocumentChunk]] = []
        for chunk in self.chunks:
            chunk_tokens = self._keyword_tokens(chunk.content)
            overlap = len(query_tokens & chunk_tokens)
            frequency = sum(chunk.content.lower().count(token) for token in query_tokens)
            score = overlap * 10 + frequency
            scored.append((score, -chunk.index, chunk))

        ranked = [item for item in scored if item[0] > 0]
        if not ranked:
            return self.chunks[: self.max_context_chunks]

        selected = sorted(
            ranked,
            key=lambda item: (-item[0], item[2].index),
        )[: self.max_context_chunks]
        return sorted((item[2] for item in selected), key=lambda chunk: chunk.index)

    def _grouped_chunks_for_global_tasks(self) -> list[DocumentChunk]:
        if len(self.chunks) <= self.max_global_analysis_chunks:
            return self.chunks

        group_size = math.ceil(len(self.chunks) / self.max_global_analysis_chunks)
        grouped_chunks: list[DocumentChunk] = []

        for group_index, start in enumerate(range(0, len(self.chunks), group_size), start=1):
            chunk_group = self.chunks[start : start + group_size]
            start_label = chunk_group[0].index
            end_label = chunk_group[-1].index
            label = f"Sections {start_label}-{end_label}"
            content = "\n\n".join(chunk.content for chunk in chunk_group)
            grouped_chunks.append(
                DocumentChunk(
                    index=group_index,
                    label=label,
                    content=content,
                    word_count=len(content.split()),
                )
            )

        return grouped_chunks

    def _format_chunks(self, chunks: list[DocumentChunk]) -> str:
        return "\n\n".join(f"{chunk.label}\n{chunk.content}" for chunk in chunks)

    def _keyword_tokens(self, text: str) -> set[str]:
        tokens = re.findall(r"[a-zA-Z0-9]{3,}", text.lower())
        return {token for token in tokens if token not in STOPWORDS}

    def _remember(self, user_prompt: str, answer: str) -> None:
        self.history.extend(
            [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": answer},
            ]
        )
        self._compact_history()
        self._persist_memory()

    def _compact_history(self) -> None:
        max_messages = self.max_recent_turns * 2
        while len(self.history) > max_messages:
            oldest_user = self.history.pop(0)
            oldest_assistant = self.history.pop(0)
            self.memory_notes.append(
                self._summarize_turn(
                    oldest_user.get("content", ""),
                    oldest_assistant.get("content", ""),
                )
            )

        if len(self.memory_notes) > self.max_memory_notes:
            self.memory_notes = self.memory_notes[-self.max_memory_notes :]

    def _summarize_turn(self, user_text: str, assistant_text: str) -> str:
        question = self._clip_text(user_text, 140)
        answer = self._clip_text(assistant_text, 220)
        return f"User asked: {question} | Assistant answered: {answer}"

    def _clip_text(self, value: str, limit: int) -> str:
        normalized = " ".join(value.split())
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 3].rstrip() + "..."

    def _build_session_id(self) -> str:
        digest = hashlib.sha256(
            f"{self.document_name}\n{self.document}".encode("utf-8")
        ).hexdigest()
        return digest[:24]

    def _load_memory(self) -> None:
        store = self._load_store()
        session_data = store.get("sessions", {}).get(self.session_id)
        if not session_data:
            return

        self.history = self._sanitize_messages(session_data.get("history", []))
        if len(self.history) % 2 != 0:
            self.history = self.history[1:]
        max_messages = self.max_recent_turns * 2
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]

        self.memory_notes = [
            str(note)
            for note in session_data.get("memory_notes", [])
            if str(note).strip()
        ][-self.max_memory_notes :]

        self.restored_turns = len(self.history) // 2
        self.restored_notes = len(self.memory_notes)

    def _persist_memory(self) -> None:
        store = self._load_store()
        store.setdefault("sessions", {})[self.session_id] = {
            "document_name": self.document_name,
            "history": self.history,
            "memory_notes": self.memory_notes,
        }
        self._save_store(store)

    def _load_store(self) -> dict[str, Any]:
        if not self.memory_store.exists():
            return {"sessions": {}}

        try:
            with self.memory_store.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except (OSError, json.JSONDecodeError):
            return {"sessions": {}}

        if not isinstance(data, dict):
            return {"sessions": {}}

        sessions = data.get("sessions")
        if not isinstance(sessions, dict):
            data["sessions"] = {}
        return data

    def _save_store(self, store: dict[str, Any]) -> None:
        self.memory_store.parent.mkdir(parents=True, exist_ok=True)
        with self.memory_store.open("w", encoding="utf-8") as file:
            json.dump(store, file, indent=2)

    def _sanitize_messages(
        self, messages: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        clean_messages: list[dict[str, str]] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            content = message.get("content")
            if role not in {"user", "assistant"}:
                continue
            if not isinstance(content, str) or not content.strip():
                continue
            clean_messages.append({"role": role, "content": content})
        return clean_messages

    def _normalize_graph_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        title = str(payload.get("title") or "Document Knowledge Graph").strip()
        summary = str(
            payload.get("summary")
            or "High-level relationships extracted from the current document."
        ).strip()

        nodes_raw = payload.get("nodes")
        edges_raw = payload.get("edges")
        if not isinstance(nodes_raw, list) or not isinstance(edges_raw, list):
            raise RuntimeError("The graph response was missing nodes or edges.")

        nodes: list[dict[str, str]] = []
        node_ids: set[str] = set()
        for raw_node in nodes_raw:
            if not isinstance(raw_node, dict):
                continue
            label = str(raw_node.get("label") or raw_node.get("id") or "").strip()
            if not label:
                continue
            node_id = self._slugify(str(raw_node.get("id") or label))
            if node_id in node_ids:
                continue
            group = str(raw_node.get("group") or "concept").strip() or "concept"
            node_ids.add(node_id)
            nodes.append({"id": node_id, "label": label, "group": group})

        if not nodes:
            raise RuntimeError("The graph response did not contain any nodes.")

        edges: list[dict[str, str]] = []
        for raw_edge in edges_raw:
            if not isinstance(raw_edge, dict):
                continue
            source = self._slugify(str(raw_edge.get("source") or "").strip())
            target = self._slugify(str(raw_edge.get("target") or "").strip())
            label = str(raw_edge.get("label") or "related to").strip() or "related to"
            if not source or not target:
                continue
            if source not in node_ids or target not in node_ids:
                continue
            edges.append({"source": source, "target": target, "label": label})

        return {
            "title": title,
            "summary": summary,
            "nodes": nodes,
            "edges": edges,
        }

    def _slugify(self, value: str) -> str:
        value = value.lower().strip()
        value = re.sub(r"[^a-z0-9]+", "-", value)
        return value.strip("-")


def create_document_session(
    document: str,
    document_name: str = "document",
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    max_recent_turns: int = DEFAULT_RECENT_TURNS,
) -> DocumentSession:
    return DocumentSession(
        document=document,
        document_name=document_name,
        api_key=api_key,
        model=model,
        max_recent_turns=max_recent_turns,
    )


def answer_question(
    document: str,
    question: str,
    task_type: int = 1,
    api_key: str | None = None,
) -> str:
    session = create_document_session(document=document, api_key=api_key)
    if task_type == 1:
        return session.ask(question)
    if task_type == 2:
        return session.generate_summary(question)
    if task_type == 3:
        return session.extract_key_information(question)
    if task_type == 4:
        return session.visualize_as_graph(question)
    if task_type == 5:
        return session.provide_feedback(question)
    if task_type == 6:
        return session.generate_research_brief(question)
    if task_type == 7:
        return session.extract_action_items(question)
    return "Invalid type. Please choose a valid option."
