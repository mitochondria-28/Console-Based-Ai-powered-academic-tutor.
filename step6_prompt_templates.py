"""
step6_prompt_templates.py
----------------------------
STEP 6: Prompt Templates

Requirements covered:
- Create reusable templates for: Explanations, Quiz generation,
  Revision notes, Study planning
- Allow dynamic input substitution

This step formalizes what we were doing ad-hoc in earlier steps into a
single, reusable PROMPT_LIBRARY. Any part of the app can pull a template
by name and fill in variables -- this is what makes the system "modular
and scalable" per the assignment's Step 10 requirement.
"""

from langchain_core.prompts import PromptTemplate
from config import get_llm

llm = get_llm()

# A central library of reusable templates, keyed by purpose.
# {variable} placeholders allow dynamic substitution at call time.
PROMPT_LIBRARY = {
    "explanation": PromptTemplate.from_template(
        "Explain '{topic}' at a {level} level. Length: {length}."
    ),
    "quiz": PromptTemplate.from_template(
        "Create a {num_questions}-question {difficulty} quiz on '{topic}'. "
        "Use multiple-choice format with 4 options and mark the correct answer."
    ),
    "revision_notes": PromptTemplate.from_template(
        "Create concise revision notes on '{topic}' formatted as bullet points, "
        "covering: {focus_areas}."
    ),
    "study_plan": PromptTemplate.from_template(
        "Create a {duration}-day study plan for a student preparing for an exam "
        "on '{subject}'. Daily goals should be specific and achievable. "
        "Student's current level: {current_level}."
    ),
}


def run_template(template_name: str, **kwargs) -> str:
    """
    Look up a template by name, substitute the dynamic inputs, and run it.
    Example: run_template("quiz", topic="Cell Biology", num_questions=5, difficulty="medium")
    """
    if template_name not in PROMPT_LIBRARY:
        raise ValueError(f"No template named '{template_name}'. Available: {list(PROMPT_LIBRARY)}")

    template = PROMPT_LIBRARY[template_name]
    chain = template | llm
    return chain.invoke(kwargs).content


def demo():
    print("=== STEP 6: Prompt Templates Demo ===\n")

    print(">> Explanation Template")
    print(run_template("explanation", topic="Object-Oriented Programming", level="beginner", length="one paragraph"))

    print("\n>> Quiz Template")
    print(run_template("quiz", topic="World War II", num_questions=3, difficulty="medium"))

    print("\n>> Revision Notes Template")
    print(run_template("revision_notes", topic="Database Normalization", focus_areas="1NF, 2NF, 3NF, BCNF"))

    print("\n>> Study Plan Template")
    print(run_template("study_plan", duration=5, subject="Data Structures & Algorithms", current_level="intermediate"))


if __name__ == "__main__":
    demo()
