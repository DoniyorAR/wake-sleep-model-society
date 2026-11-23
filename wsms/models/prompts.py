"""Prompts of the reader, the closed-book questions and the support check."""

SYSTEM = ("You are a personal assistant with long-term memory of the people the user knows. "
          "Answer questions briefly and factually. If you do not know the answer, say that you "
          "do not know.")
CHECKER_SYSTEM = "You are a careful fact checker."


def closed_book(question):
    return f"Question: {question}\nAnswer in a few words."


def open_book(source_text, question):
    return f"{source_text}\n\nQuestion: {question}\nAnswer in a few words."


def support_prompt(source_text, question, answer):
    return (f"Source text:\n{source_text}\n\nQuestion: {question}\nProposed answer: {answer}\n\n"
            "Is this answer stated directly in the source text about the exact person or thing the "
            "question asks about? Reply 'no' if the text only suggests it, or states it about a "
            "different person, place or thing. Reply with only 'yes' or 'no'.")
