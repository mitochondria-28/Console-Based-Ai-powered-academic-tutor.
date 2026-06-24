# AI-Powered Academic Tutor

A console-based AI tutor built with LangChain + Google Gemini, implementing
all 10 steps of the assignment.

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Get a **free** Gemini API key (no credit card required):
   - Go to https://aistudio.google.com/app/apikey
   - Sign in with Google, click "Create API key"

3. Copy `.env.example` to `.env` and paste your key in:
   ```
   GOOGLE_API_KEY=your_key_here
   ```

4. Run the full application:
   ```
   python main.py
   ```

   Or run any individual step's demo directly to test/screenshot it on its own:
   ```
   python step1_api_integration.py
   python step2_zero_shot.py
   python step3_few_shot.py
   python step4_chain_of_thought.py
   python step5_role_prompting.py
   python step6_prompt_templates.py
   python step7_langchain_chains.py
   python step8_memory_system.py
   python step9_agents_tools.py
   ```

## File-to-Step Mapping (for your report's Implementation Details section)

| File                          | Assignment Step              |
|-------------------------------|-------------------------------|
| `config.py`                   | Shared setup (model/API key) |
| `step1_api_integration.py`    | Step 1 - LLM API Integration |
| `step2_zero_shot.py`          | Step 2 - Zero-Shot Prompting |
| `step3_few_shot.py`           | Step 3 - Few-Shot Prompting  |
| `step4_chain_of_thought.py`   | Step 4 - Chain-of-Thought    |
| `step5_role_prompting.py`     | Step 5 - Role Prompting      |
| `step6_prompt_templates.py`   | Step 6 - Prompt Templates    |
| `step7_langchain_chains.py`   | Step 7 - LangChain Chains    |
| `step8_memory_system.py`      | Step 8 - Memory System       |
| `step9_agents_tools.py`       | Step 9 - Agents & Tools      |
| `main.py`                     | Step 10 - Final Integration  |

## Notes for your report

- **Architecture**: `main.py` is the entry point. It imports every step module
  and routes user menu choices to them, sharing one `TutorSession` (chat memory)
  and one `StudentMemory` (learning history) across the whole run — this is your
  "module breakdown" and "smooth interaction between components" diagram material.
- **Why Gemini**: free tier, no card needed, integrates natively with LangChain
  via `langchain-google-genai`. Mention this under Limitations (rate limits:
  free tier is request-capped per minute/day — fine for a course project, not
  for production).
- **LangChain version used**: 1.x (`create_agent` for Step 9 — the current
  recommended agent API, replacing older `initialize_agent`/`AgentExecutor`
  patterns you may see in older tutorials).
- **Limitations to mention**: API rate limits on the free tier; model
  occasionally varies output formatting despite few-shot examples; no RAG/
  document grounding yet (listed as a Future Enhancement in the assignment).
- **Originality**: This code is a starting implementation — read through it,
  run it, tweak prompts/tools, and add your own touches (e.g. extra roles,
  extra tools) before submitting, since the assignment requires original code.
