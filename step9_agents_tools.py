"""
step9_agents_tools.py
------------------------
STEP 9: Agents & Tools

Requirements covered:
- Build an AI agent that uses tools: Calculator, Study Planner, Information Summarizer
- Enable multi-step reasoning and tool selection

We use LangChain 1.x's `create_agent` (the current recommended agent
constructor, built on LangGraph under the hood). The agent decides, on
its own, WHICH tool to call (or none) based on each tool's docstring --
that decision-making is the "tool selection" requirement, and chaining
multiple tool calls together for one query is the "multi-step reasoning"
requirement.
"""

import math
from langchain_core.tools import tool
from langchain.agents import create_agent
from config import GOOGLE_API_KEY, MODEL_NAME

# Make sure the API key is visible to LangChain's provider-string loader
import os
os.environ.setdefault("GOOGLE_API_KEY", GOOGLE_API_KEY or "")


@tool
def calculator(expression: str) -> str:
    """
    Evaluates a basic arithmetic/math expression and returns the numeric result.
    Use this whenever the student's question requires a numeric calculation,
    e.g. '23 * 7', 'sqrt(144)', '(15 + 9) / 3'.
    """
    try:
        allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Could not evaluate expression: {e}"


@tool
def study_planner(subject: str, days: int) -> str:
    """
    Generates a simple day-by-day study plan for a given subject and number
    of days. Use this when the student asks for a study schedule, revision
    plan, or how to organize their exam preparation time.
    """
    days = max(1, int(days))
    phases = ["Overview & fundamentals", "Core concepts in depth",
              "Practice problems", "Weak-area review", "Full revision & mock test"]
    plan_lines = []
    for d in range(1, days + 1):
        phase = phases[min(d - 1, len(phases) - 1)]
        plan_lines.append(f"Day {d}: {subject} - {phase}")
    return "\n".join(plan_lines)


@tool
def information_summarizer(text: str) -> str:
    """
    Produces a short 2-3 sentence summary of a longer piece of text.
    Use this when the student pastes a long passage and wants the key
    points condensed.
    """
    # Lightweight extractive fallback; the agent's LLM call typically
    # handles true summarization, but this tool demonstrates a deterministic
    # utility tool the agent can invoke directly.
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
    summary = ". ".join(sentences[:3])
    return summary + ("." if summary and not summary.endswith(".") else "")


TOOLS = [calculator, study_planner, information_summarizer]

agent = create_agent(
    model=f"google_genai:{MODEL_NAME}",
    tools=TOOLS,
    system_prompt=(
        "You are an AI academic tutor agent. You have access to tools: "
        "calculator, study_planner, and information_summarizer. "
        "Decide which tool(s), if any, are needed to fully answer the "
        "student's request, call them, and then give a clear final answer "
        "that incorporates the tool results."
    ),
)


def ask_agent(query: str) -> str:
    """Run the agent on a single query and return its final text answer."""
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    final_message = result["messages"][-1]
    return final_message.content


def demo():
    print("=== STEP 9: Agents & Tools Demo ===\n")

    queries = [
        "What is 144 divided by 12, and then multiply that result by 5?",
        "Make me a 4-day study plan for Organic Chemistry.",
        "Summarize this: The water cycle describes how water moves through "
        "Earth's atmosphere, land, and oceans. It begins with evaporation, "
        "where heat from the sun turns water into vapor. This vapor rises and "
        "cools, forming clouds through condensation. Eventually, water falls "
        "back to Earth as precipitation, completing the cycle.",
    ]

    for q in queries:
        print(f">> Query: {q}")
        print("Agent:", ask_agent(q))
        print("-" * 60)


if __name__ == "__main__":
    demo()
