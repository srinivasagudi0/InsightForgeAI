from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from intel import DocumentSession


class FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, model, messages, response_format=None):
        self.calls.append(
            {
                "model": model,
                "messages": messages,
                "response_format": response_format,
            }
        )
        prompt = messages[-1]["content"]
        if response_format == {"type": "json_object"}:
            content = json.dumps(
                {
                    "title": "Mock Graph",
                    "summary": "Mocked graph output.",
                    "nodes": [
                        {"id": "paper", "label": "Paper", "group": "source"},
                        {"id": "result", "label": "Result", "group": "finding"},
                    ],
                    "edges": [
                        {"source": "paper", "target": "result", "label": "reports"}
                    ],
                }
            )
        else:
            content = f"mocked response for: {prompt}"
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


class FakeClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletions())


def test_large_document_chunking_and_summary(tmp_path: Path) -> None:
    document = "\n\n".join(
        [
            f"Section {index} discusses transformers, retrieval, and evaluation details. "
            f"Chunk marker {index}."
            for index in range(1, 15)
        ]
    )
    session = DocumentSession(
        document=document,
        document_name="paper.txt",
        api_key="test-key",
        memory_store=tmp_path / "memory.json",
        inline_word_limit=50,
        chunk_word_target=25,
        max_global_analysis_chunks=5,
    )
    session.client = FakeClient()

    answer = session.generate_summary("evaluation")

    assert "mocked response for:" in answer
    assert len(session.chunks) > 1
    assert session.analysis_cache
    assert len(session.client.chat.completions.calls) > 1


def test_question_context_selects_relevant_chunks(tmp_path: Path) -> None:
    document = "\n\n".join(
        [
            "This chunk covers biology and cell growth.",
            "This chunk covers robotics and motion planning.",
            "This chunk covers astronomy and telescopes.",
        ]
    )
    session = DocumentSession(
        document=document,
        document_name="paper.txt",
        api_key="test-key",
        memory_store=tmp_path / "memory.json",
        inline_word_limit=5,
        chunk_word_target=8,
        max_context_chunks=1,
    )
    session.client = FakeClient()

    context = session._context_for_question("What does it say about robotics?")
    assert "robotics" in context.lower()
    assert "astronomy" not in context.lower()


def test_graph_data_normalization_and_memory_persistence(tmp_path: Path) -> None:
    memory_store = tmp_path / "memory.json"
    session = DocumentSession(
        document="A document about robotics results.",
        document_name="robotics.txt",
        api_key="test-key",
        memory_store=memory_store,
    )
    session.client = FakeClient()

    graph = session.build_graph_data("results")

    assert graph["title"] == "Mock Graph"
    assert len(graph["nodes"]) == 2

    reloaded = DocumentSession(
        document="A document about robotics results.",
        document_name="robotics.txt",
        api_key="test-key",
        memory_store=memory_store,
    )
    assert reloaded.has_restored_memory()
