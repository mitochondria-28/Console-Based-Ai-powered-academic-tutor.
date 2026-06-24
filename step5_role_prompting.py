"""
step5_role_prompting.py
--------------------------
STEP 5: Role Prompting

Requirements covered:
- Implement multiple tutor roles: Teacher, Examiner, Study Coach, Subject Expert
- Modify responses based on selected role

Role prompting works by changing the SYSTEM message, which sets the
"persona" / behavior constraints the model should adopt for the entire
conversation, before the user's actual question is ever sent.
"""

from langchain_core.prompts import ChatPromptTemplate
from config import get_llm

llm = get_llm()

# Each role has a distinct system instruction that changes tone, focus,
# and structure of the response -- this IS the "modify responses based
# on selected role" requirement.
ROLE_DEFINITIONS = {
    "Teacher": (
        "You are a patient, friendly Teacher. Explain concepts clearly with "
        "simple analogies, check understanding, and encourage the student."
    ),
    "Examiner": (
        "You are a strict Examiner. Respond formally, focus on precision and "
        "completeness, point out gaps or errors in reasoning, and grade answers "
        "as if marking an exam."
    ),
    "Study Coach": (
        "You are a motivational Study Coach. Focus on study strategies, time "
        "management, and encouragement rather than deep technical detail."
    ),
    "Subject Expert": (
        "You are a Subject Expert (PhD-level). Give technically rigorous, "
        "in-depth answers using correct terminology, assuming an advanced audience."
    ),
}


def ask_as_role(role: str, query: str) -> str:
    """Send a query to the LLM using the chosen role's system persona."""
    if role not in ROLE_DEFINITIONS:
        raise ValueError(f"Unknown role '{role}'. Choose from: {list(ROLE_DEFINITIONS)}")

    prompt = ChatPromptTemplate.from_messages([
        ("system", ROLE_DEFINITIONS[role]),
        ("human", "{query}"),
    ])
    chain = prompt | llm
    return chain.invoke({"query": query}).content


def demo():
    print("=== STEP 5: Role Prompting Demo ===\n")
    question = "Explain the significance of the French Revolution."

    for role in ROLE_DEFINITIONS:
        print(f">> Role: {role}")
        print(ask_as_role(role, question))
        print("-" * 60)


if __name__ == "__main__":
    demo()
