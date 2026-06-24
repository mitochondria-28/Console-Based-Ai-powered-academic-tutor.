"""
config.py
---------
Centralized configuration for the AI Academic Tutor project.

Loads the Gemini API key from a .env file (never hardcode it / never commit it)
and exposes a ready-to-use ChatGoogleGenerativeAI model instance that every
other module imports.

Why Gemini?
- Google AI Studio gives a genuinely free API key (no credit card).
- The free tier covers Gemini 2.5 Flash / Flash-Lite, which is more than
  enough for an academic tutor project.
- It plugs into LangChain via the official `langchain-google-genai` package.

How to get a free key:
1. Go to https://aistudio.google.com/app/apikey
2. Sign in with a Google account and click "Create API key"
3. Copy the key into a file named `.env` in this folder:
       GOOGLE_API_KEY=your_key_here
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load variables from .env into the environment
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY not found. Create a .env file in this folder with:\n"
        "GOOGLE_API_KEY=your_key_here\n"
        "Get a free key at https://aistudio.google.com/app/apikey"
    )

# Model name kept in one place so it's easy to swap later
MODEL_NAME = "gemini-2.5-flash"


def get_llm(temperature: float = 0.4) -> ChatGoogleGenerativeAI:
    """
    Factory function that returns a configured Gemini chat model.
    Every module calls this instead of constructing the model itself,
    so the whole project stays consistent and easy to reconfigure.

    `.with_retry()` makes the model automatically retry on transient
    errors (e.g. Google's "503 UNAVAILABLE - model overloaded" during
    high-demand periods, or occasional 429 rate-limit hits) instead of
    crashing the whole app. It waits with exponential backoff between
    attempts before giving up.
    """
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=GOOGLE_API_KEY,
        temperature=temperature,
    )
    return llm.with_retry(
        stop_after_attempt=4,
        wait_exponential_jitter=True,
    )