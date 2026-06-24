"""
step4_chain_of_thought.py
---------------------------
STEP 4: Chain-of-Thought (CoT) Prompting

Requirements covered:
- Solve academic and reasoning-based problems step-by-step
- Break complex problems into logical steps
- Improve clarity of solutions

Chain-of-Thought prompting explicitly instructs the model to "think out
loud" through intermediate reasoning steps before giving a final answer.
This is critical for math, physics, and logic problems where jumping
straight to an answer often produces mistakes.
"""

from langchain_core.prompts import PromptTemplate
from config import get_llm

llm = get_llm()

cot_prompt = PromptTemplate.from_template(
    """You are solving an academic problem. Think through it step-by-step:

1. First, identify what is being asked and what information is given.
2. Break the problem into smaller logical steps.
3. Solve each step one at a time, showing your reasoning and any formulas used.
4. Combine the steps into a clear final answer, clearly labeled "Final Answer:".

Problem: {problem}

Step-by-step solution:"""
)


def solve_with_cot(problem: str) -> str:
    chain = cot_prompt | llm
    return chain.invoke({"problem": problem}).content


def demo():
    print("=== STEP 4: Chain-of-Thought Prompting Demo ===\n")

    problems = [
        "A train travels 180 km in 3 hours. If it then travels another 120 km "
        "at the same speed, how long does the second leg take?",
        "If a substance has a half-life of 10 years and you start with 80 grams, "
        "how much remains after 30 years?",
    ]

    for p in problems:
        print(f">> Problem: {p}")
        print(solve_with_cot(p))
        print("-" * 60)


if __name__ == "__main__":
    demo()
