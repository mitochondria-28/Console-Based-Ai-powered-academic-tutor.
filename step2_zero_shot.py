"""
step2_zero_shot.py
-------------------
STEP 2: Zero-Shot Prompting

Requirements covered:
- Generate explanations for academic topics
- Summarize study material without examples
- Simplify complex concepts for students

"Zero-shot" means we give the model an instruction with NO examples of
input/output pairs -- it must rely purely on its own pretrained knowledge
to follow the instruction correctly.
"""

from langchain_core.prompts import PromptTemplate
from config import get_llm

llm = get_llm()

# --- 1. Explanation ---
explain_prompt = PromptTemplate.from_template(
    "Explain the academic topic '{topic}' clearly and accurately, "
    "suitable for a university-level student. Keep it to one well-structured paragraph."
)

# --- 2. Summarization ---
summarize_prompt = PromptTemplate.from_template(
    "Summarize the following study material into concise key points:\n\n{text}"
)

# --- 3. Simplification ---
simplify_prompt = PromptTemplate.from_template(
    "Simplify the following concept so a complete beginner with no background "
    "in the subject can understand it. Use plain, everyday language:\n\n{concept}"
)


def explain_topic(topic: str) -> str:
    chain = explain_prompt | llm
    return chain.invoke({"topic": topic}).content


def summarize_text(text: str) -> str:
    chain = summarize_prompt | llm
    return chain.invoke({"text": text}).content


def simplify_concept(concept: str) -> str:
    chain = simplify_prompt | llm
    return chain.invoke({"concept": concept}).content


def demo():
    print("=== STEP 2: Zero-Shot Prompting Demo ===\n")

    print(">> Explanation Task")
    print(explain_topic("the process of photosynthesis"))

    print("\n>> Summarization Task")
    sample_text = (
        "The mitochondria is the powerhouse of the cell. It generates most of the "
        "cell's supply of adenosine triphosphate (ATP), used as a source of chemical "
        "energy. Mitochondria have a double membrane structure and their own DNA, "
        "supporting the theory that they originated from ancient bacteria that were "
        "engulfed by early eukaryotic cells (the endosymbiotic theory)."
    )
    print(summarize_text(sample_text))

    print("\n>> Simplification Task")
    print(simplify_concept("Quantum entanglement"))


if __name__ == "__main__":
    demo()
