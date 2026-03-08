#main.py
from __future__ import annotations

import os
from typing import Callable

from intel import DocumentSession, create_document_session


def main() -> None:
    greet()

    while True:
        file_path = input("Enter the path to the document: ").strip()
        if validate_file_path(file_path):
            break

    document = get_file(file_path)

    try:
        session = create_document_session(
            document=document,
            document_name=os.path.basename(file_path) or file_path,
        )
    except RuntimeError as exc:
        print(f"Setup error: {exc}")
        return

    print("Document loaded successfully!")
    if session.has_restored_memory():
        print(
            f"Restored memory for this document: {session.restored_turns} recent turns and {session.restored_notes} saved notes."
        )

    while True:
        show_menu()
        choice = input("Enter the number corresponding to your choice: ").strip()

        if choice == "1":
            if not chat_mode(session):
                break
        elif choice == "2":
            focus = prompt_optional_focus("What should the summary focus on?")
            run_and_print("Summary", lambda: session.generate_summary(focus))
        elif choice == "3":
            focus = prompt_optional_focus("What key information should be prioritized?")
            run_and_print(
                "Key Information", lambda: session.extract_key_information(focus)
            )
        elif choice == "4":
            focus = prompt_optional_focus("What relationships should the graph emphasize?")
            run_and_print("Graph View", lambda: session.visualize_as_graph(focus))
        elif choice == "5":
            focus = prompt_optional_focus("What kind of feedback do you want?")
            run_and_print("Feedback", lambda: session.provide_feedback(focus))
        elif choice == "6":
            session.clear_memory()
            print("Saved memory for this document has been cleared.")
        elif choice == "7":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")


def greet() -> None:
    print(
        "Hello! I am InsightForge-AI, your personal assistant for analyzing and summarizing documents."
    )
    print("Please provide the path to the document you want me to analyze.")


def show_menu() -> None:
    print("\nWhat would you like to do with the document?")
    print("1. Chat with the document")
    print("2. Generate a summary of the document")
    print("3. Extract key information from the document")
    print("4. Visualize the document as a graph")
    print("5. Provide feedback on the document")
    print("6. Clear saved memory for this document")
    print("7. Exit")


def chat_mode(session: DocumentSession) -> bool:
    print("\nChat mode started.")
    print("Type /menu to return to the main menu, /clear to reset memory, or /exit to quit.")

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() == "/menu":
            return True
        if question.lower() == "/clear":
            session.clear_memory()
            print("InsightForge-AI: Memory cleared for this document.")
            continue
        if question.lower() == "/exit":
            print("Goodbye!")
            return False

        try:
            answer = session.ask(question)
        except RuntimeError as exc:
            print(f"InsightForge-AI: {exc}")
            continue

        print(f"InsightForge-AI: {answer}")


def run_and_print(title: str, action: Callable[[], str]) -> None:
    try:
        result = action()
    except RuntimeError as exc:
        print(f"{title}: {exc}")
        return

    print(f"\n{title}:\n{result}")


def prompt_optional_focus(prompt: str) -> str:
    return input(f"{prompt} Press Enter for a general response: ").strip()


def validate_file_path(file_path: str) -> bool:
    if not os.path.isfile(file_path):
        print("The provided file path does not exist. Please try again.")
        return False
    return True


def get_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        return file.read()


if __name__ == "__main__":
    main()
