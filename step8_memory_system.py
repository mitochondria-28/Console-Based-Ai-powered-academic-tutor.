"""
step8_memory_system.py
-------------------------
STEP 8: Memory System

Requirements covered:
- Store user identity and learning history
- Maintain context across conversations
- Enable personalized responses

Note for the report: as of LangChain 1.x, the old `ConversationBufferMemory`
/ `ConversationSummaryMemory` classes from earlier LangChain versions are
deprecated in favor of explicit, application-managed state (LangChain's docs
now point to LangGraph-style persistence). For a console app like this, a
simple JSON-backed memory store is clearer to implement, easier to explain
in a report, and gives full control over what's remembered -- so that's
what we build here.
"""

import json
import os
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from config import get_llm

llm = get_llm()
MEMORY_FILE = "student_memory.json"


class StudentMemory:
    """
    Persistent memory for a single student across sessions.
    Stores identity (name, level) and a running learning history
    (topics studied, quiz performance, notes) to a local JSON file.
    """

    def __init__(self, student_name: str, memory_path: str = MEMORY_FILE):
        self.memory_path = memory_path
        self.student_name = student_name
        self.data = self._load()

        if "name" not in self.data:
            self.data["name"] = student_name
            self.data["level"] = "unspecified"
            self.data["topics_studied"] = []
            self.data["history"] = []
            self._save()

    def _load(self) -> dict:
        if os.path.exists(self.memory_path):
            with open(self.memory_path, "r") as f:
                all_students = json.load(f)
        else:
            all_students = {}
        return all_students.get(self.student_name, {})

    def _save(self):
        if os.path.exists(self.memory_path):
            with open(self.memory_path, "r") as f:
                all_students = json.load(f)
        else:
            all_students = {}
        all_students[self.student_name] = self.data
        with open(self.memory_path, "w") as f:
            json.dump(all_students, f, indent=2)

    def set_level(self, level: str):
        self.data["level"] = level
        self._save()

    def log_topic(self, topic: str, summary: str):
        """Record that this topic was studied, with a timestamp and short summary."""
        self.data["topics_studied"].append(topic)
        self.data["history"].append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "topic": topic,
            "summary": summary,
        })
        self._save()

    def get_context_string(self) -> str:
        """Build a short text summary of this student's history, to inject into prompts."""
        topics = ", ".join(self.data.get("topics_studied", [])) or "none yet"
        return (
            f"Student name: {self.data.get('name')}. "
            f"Level: {self.data.get('level')}. "
            f"Topics studied so far: {topics}."
        )


def personalized_response(memory: StudentMemory, query: str) -> str:
    """
    Generate a response that takes the student's stored identity and
    learning history into account -- this is the 'personalization' requirement.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a personal AI tutor. Use the student's known background to "
         "tailor your response (e.g. reference earlier topics, match their level). "
         "Student context: {context}"),
        ("human", "{query}"),
    ])
    chain = prompt | llm
    return chain.invoke({"context": memory.get_context_string(), "query": query}).content


def demo():
    print("=== STEP 8: Memory System Demo ===\n")

    memory = StudentMemory(student_name="Kusu")
    memory.set_level("undergraduate, final year")

    print(">> First interaction (no prior topics)")
    r1 = personalized_response(memory, "Explain gradient descent.")
    print(r1)
    memory.log_topic("Gradient Descent", r1[:150])

    print("\n>> Second interaction (should reference earlier topic)")
    r2 = personalized_response(memory, "How does that relate to training a neural network?")
    print(r2)
    memory.log_topic("Neural Network Training", r2[:150])

    print("\n>> Stored memory file contents:")
    with open(MEMORY_FILE, "r") as f:
        print(f.read())


if __name__ == "__main__":
    demo()
