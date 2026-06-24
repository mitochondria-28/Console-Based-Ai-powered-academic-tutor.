"""
step3_few_shot.py
-------------------
STEP 3: Few-Shot Prompting

Requirements covered:
- Use examples to structure outputs
- Generate quizzes and categorized answers
- Improve formatting consistency of responses

"Few-shot" means we show the model 2-3 worked examples of the exact
input -> output pattern we want BEFORE giving it the real task. This
dramatically improves output consistency/formatting compared to zero-shot.
"""

from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from config import get_llm

llm = get_llm()

# ---------------------------------------------------------------
# 3.1 Few-shot quiz generation -> teaches the model the EXACT format
# ---------------------------------------------------------------
quiz_examples = [
    {
        "topic": "Photosynthesis",
        "quiz": (
            "Q1: What gas do plants absorb during photosynthesis?\n"
            "A) Oxygen  B) Carbon Dioxide  C) Nitrogen  D) Hydrogen\n"
            "Answer: B\n\n"
            "Q2: Where does photosynthesis mainly take place in a plant cell?\n"
            "A) Nucleus  B) Mitochondria  C) Chloroplast  D) Ribosome\n"
            "Answer: C"
        ),
    },
    {
        "topic": "Newton's Laws of Motion",
        "quiz": (
            "Q1: Which law states that an object in motion stays in motion unless acted on by a force?\n"
            "A) First Law  B) Second Law  C) Third Law  D) Law of Gravitation\n"
            "Answer: A\n\n"
            "Q2: F = ma represents which law?\n"
            "A) First Law  B) Second Law  C) Third Law  D) None\n"
            "Answer: B"
        ),
    },
]

quiz_example_template = PromptTemplate.from_template(
    "Topic: {topic}\nQuiz:\n{quiz}"
)

quiz_few_shot_prompt = FewShotPromptTemplate(
    examples=quiz_examples,
    example_prompt=quiz_example_template,
    prefix=(
        "You are a quiz generator. Follow the EXACT format shown in the examples "
        "below: 2 multiple-choice questions, each with 4 options (A-D) and an "
        "'Answer:' line."
    ),
    suffix="Topic: {topic}\nQuiz:",
    input_variables=["topic"],
)


def generate_quiz(topic: str) -> str:
    chain = quiz_few_shot_prompt | llm
    return chain.invoke({"topic": topic}).content


# ---------------------------------------------------------------
# 3.2 Few-shot categorized answers -> teaches consistent category labels
# ---------------------------------------------------------------
category_examples = [
    {
        "question": "What causes rust to form on iron?",
        "answer": "Category: Chemistry\nAnswer: Rust forms when iron reacts with oxygen and moisture in the air, a process called oxidation.",
    },
    {
        "question": "Why do we have seasons?",
        "answer": "Category: Earth Science\nAnswer: Seasons occur because Earth's axis is tilted, changing how directly sunlight hits different regions throughout the year.",
    },
]

category_example_template = PromptTemplate.from_template(
    "Question: {question}\n{answer}"
)

category_few_shot_prompt = FewShotPromptTemplate(
    examples=category_examples,
    example_prompt=category_example_template,
    prefix=(
        "Answer each academic question and classify it into a subject category, "
        "EXACTLY matching the format of the examples below."
    ),
    suffix="Question: {question}",
    input_variables=["question"],
)


def categorized_answer(question: str) -> str:
    chain = category_few_shot_prompt | llm
    return chain.invoke({"question": question}).content


def demo():
    print("=== STEP 3: Few-Shot Prompting Demo ===\n")

    print(">> Few-shot Quiz Generation")
    print(generate_quiz("The Water Cycle"))

    print("\n>> Few-shot Categorized Answer")
    print(categorized_answer("Why does the moon have phases?"))


if __name__ == "__main__":
    demo()
