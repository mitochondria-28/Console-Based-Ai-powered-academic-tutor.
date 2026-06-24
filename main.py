"""
main.py
--------
STEP 10: Final Integration

Requirements covered:
- Combine all modules into a single console application
- Ensure smooth interaction between all components
- Maintain modular and scalable design

This is the entry point a student runs (`python main.py`). It presents
a menu that routes to every module built in Steps 1-9, while keeping
ONE shared StudentMemory instance alive across the whole session so
personalization (Step 8) works across every feature used.
"""

from step1_api_integration import TutorSession
from step2_zero_shot import explain_topic, summarize_text, simplify_concept
from step3_few_shot import generate_quiz, categorized_answer
from step4_chain_of_thought import solve_with_cot
from step5_role_prompting import ask_as_role, ROLE_DEFINITIONS
from step6_prompt_templates import run_template
from step7_langchain_chains import run_topic_to_quiz_pipeline
from step8_memory_system import StudentMemory, personalized_response
from step9_agents_tools import ask_agent


MENU = """
====================================================
   AI-POWERED ACADEMIC TUTOR - MAIN MENU
====================================================
 1. Chat with Tutor (session memory)         [Step 1]
 2. Explain / Summarize / Simplify a topic    [Step 2]
 3. Generate a Quiz (few-shot)                [Step 3]
 4. Step-by-step Problem Solver (CoT)         [Step 4]
 5. Talk to a specific Tutor Role             [Step 5]
 6. Use a Prompt Template                     [Step 6]
 7. Run Topic->Explanation->Notes->Quiz Chain  [Step 7]
 8. Personalized response (learning history)  [Step 8]
 9. Ask the Agent (calculator/planner/summarizer) [Step 9]
 0. Exit
====================================================
"""


def menu_chat(session: TutorSession):
    query = input("Ask the tutor: ")
    print("\nTutor:", session.ask(query), "\n")


def menu_zero_shot():
    print("a) Explain  b) Summarize  c) Simplify")
    choice = input("Choose (a/b/c): ").strip().lower()
    if choice == "a":
        print(explain_topic(input("Topic: ")))
    elif choice == "b":
        print(summarize_text(input("Paste text: ")))
    elif choice == "c":
        print(simplify_concept(input("Concept: ")))
    else:
        print("Invalid choice.")


def menu_few_shot():
    print("a) Generate quiz  b) Categorized answer")
    choice = input("Choose (a/b): ").strip().lower()
    if choice == "a":
        print(generate_quiz(input("Topic: ")))
    elif choice == "b":
        print(categorized_answer(input("Question: ")))
    else:
        print("Invalid choice.")


def menu_cot():
    print(solve_with_cot(input("Describe the problem: ")))


def menu_roles():
    print("Available roles:", ", ".join(ROLE_DEFINITIONS))
    role = input("Choose a role: ").strip()
    query = input("Your question: ")
    print(ask_as_role(role, query))


def menu_templates():
    print("Available templates: explanation, quiz, revision_notes, study_plan")
    name = input("Template name: ").strip()
    if name == "explanation":
        print(run_template("explanation", topic=input("Topic: "), level=input("Level: "), length=input("Length: ")))
    elif name == "quiz":
        print(run_template("quiz", topic=input("Topic: "), num_questions=input("Num questions: "), difficulty=input("Difficulty: ")))
    elif name == "revision_notes":
        print(run_template("revision_notes", topic=input("Topic: "), focus_areas=input("Focus areas: ")))
    elif name == "study_plan":
        print(run_template("study_plan", duration=input("Duration (days): "), subject=input("Subject: "), current_level=input("Current level: ")))
    else:
        print("Unknown template.")


def menu_chain():
    result = run_topic_to_quiz_pipeline(input("Topic: "))
    print("\n--- Explanation ---\n", result["explanation"])
    print("\n--- Notes ---\n", result["notes"])
    print("\n--- Quiz ---\n", result["quiz"])


def menu_memory(memory: StudentMemory):
    query = input("Ask something (personalized): ")
    response = personalized_response(memory, query)
    print(response)
    memory.log_topic(query[:40], response[:150])


def menu_agent():
    print(ask_agent(input("Ask the agent (math / study plan / summarize): ")))


def main():
    print("Welcome to the AI-Powered Academic Tutor.")
    student_name = input("Enter your name to begin: ").strip() or "Student"

    session = TutorSession()
    memory = StudentMemory(student_name=student_name)

    while True:
        print(MENU)
        choice = input("Select an option: ").strip()

        if choice == "1":
            menu_chat(session)
        elif choice == "2":
            menu_zero_shot()
        elif choice == "3":
            menu_few_shot()
        elif choice == "4":
            menu_cot()
        elif choice == "5":
            menu_roles()
        elif choice == "6":
            menu_templates()
        elif choice == "7":
            menu_chain()
        elif choice == "8":
            menu_memory(memory)
        elif choice == "9":
            menu_agent()
        elif choice == "0":
            print("Goodbye! Keep studying.")
            break
        else:
            print("Invalid option, try again.")


if __name__ == "__main__":
    main()
