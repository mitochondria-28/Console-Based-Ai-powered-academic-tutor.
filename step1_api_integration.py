"""
step1_api_integration.py
-------------------------
STEP 1: LLM API Integration

Requirements covered:
- Connect Python application with an LLM API
- Send user queries and display responses
- Maintain session-based conversation history

This is the foundation every later step builds on. We use LangChain's
message objects (HumanMessage / AIMessage / SystemMessage) so the model
receives proper conversational context, not just a single isolated string.
"""

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from config import get_llm


class TutorSession:
    """
    Represents one conversation session with the AI tutor.
    Keeps an in-memory list of messages so the model has context
    of everything said so far in this session.
    """

    def __init__(self, system_prompt: str = "You are a helpful, encouraging academic tutor."):
        self.llm = get_llm()
        # The history always starts with a system message that sets behavior
        self.history = [SystemMessage(content=system_prompt)]

    def ask(self, user_query: str) -> str:
        """Send a query to the LLM, store it + the response in history, return the answer."""
        self.history.append(HumanMessage(content=user_query))

        response = self.llm.invoke(self.history)

        self.history.append(AIMessage(content=response.content))
        return response.content

    def show_history(self):
        """Pretty-print the conversation so far (useful for report screenshots)."""
        print("\n--- Session History ---")
        for msg in self.history:
            role = msg.__class__.__name__.replace("Message", "")
            print(f"[{role}]: {msg.content}\n")


def demo():
    print("=== STEP 1: LLM API Integration Demo ===\n")
    session = TutorSession()

    print(">> User: What is Newton's First Law of Motion?")
    print("Tutor:", session.ask("What is Newton's First Law of Motion?"))

    print("\n>> User: Can you give a real-life example of it?")
    print("Tutor:", session.ask("Can you give a real-life example of it?"))

    # This proves session memory is working: the model knows "it" refers
    # to Newton's First Law because the full history was sent.
    session.show_history()


if __name__ == "__main__":
    demo()
