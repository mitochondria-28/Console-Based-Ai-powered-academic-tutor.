"""
step7_langchain_chains.py
----------------------------
STEP 7: LangChain Chains

Requirements covered:
- Build sequential workflows: Topic -> Explanation -> Notes -> Quiz
- Ensure modular chain execution
- Pass outputs between chains

We build this with LCEL (LangChain Expression Language) -- the current
recommended way to compose chains in LangChain 1.x. Each stage is its
own small, independently testable Runnable; we then connect them so the
OUTPUT of one stage becomes the INPUT of the next.
"""

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import get_llm

llm = get_llm()
parser = StrOutputParser()

# --- Stage 1: Topic -> Explanation ---
explanation_chain = (
    PromptTemplate.from_template(
        "Explain the academic topic '{topic}' in 2-3 clear paragraphs suitable for a student."
    )
    | llm
    | parser
)

# --- Stage 2: Explanation -> Revision Notes ---
notes_chain = (
    PromptTemplate.from_template(
        "Convert the following explanation into concise bullet-point revision notes:\n\n{explanation}"
    )
    | llm
    | parser
)

# --- Stage 3: Notes -> Quiz ---
quiz_chain = (
    PromptTemplate.from_template(
        "Based on these revision notes, create a 3-question multiple-choice quiz "
        "(4 options each, mark the correct answer):\n\n{notes}"
    )
    | llm
    | parser
)


def run_topic_to_quiz_pipeline(topic: str) -> dict:
    """
    Modular sequential pipeline: Topic -> Explanation -> Notes -> Quiz.

    Each stage's output is explicitly passed as the next stage's input,
    and all intermediate outputs are returned so they can be inspected/
    screenshotted individually (per the report's 'Code Screenshots' requirement).
    """
    explanation = explanation_chain.invoke({"topic": topic})
    notes = notes_chain.invoke({"explanation": explanation})
    quiz = quiz_chain.invoke({"notes": notes})

    return {
        "topic": topic,
        "explanation": explanation,
        "notes": notes,
        "quiz": quiz,
    }



def demo():
    print("=== STEP 7: LangChain Sequential Chain Demo ===\n")
    result = run_topic_to_quiz_pipeline("The Carbon Cycle")

    print(">> Stage 1 - Explanation:\n", result["explanation"])
    print("\n>> Stage 2 - Revision Notes:\n", result["notes"])
    print("\n>> Stage 3 - Quiz:\n", result["quiz"])


if __name__ == "__main__":
    demo()
