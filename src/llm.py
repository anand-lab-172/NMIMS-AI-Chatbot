import os

from dotenv import load_dotenv

from google import genai
from openai import OpenAI
from groq import Groq

load_dotenv()

# ---------------- GEMINI ---------------- #

gemini_client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)

# ---------------- OPENAI ---------------- #

openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# ---------------- GROQ ---------------- #

groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# ---------------- DEFAULT MODEL ---------------- #

DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"

# ---------------- GEMINI ---------------- #

def ask_gemini(prompt, model_name="gemini-1.5-flash"):

    response = gemini_client.models.generate_content(
        model=model_name,
        contents=prompt
    )

    return response.text

# ---------------- CHATGPT ---------------- #

def ask_chatgpt(prompt, model_name="gpt-4o-mini"):

    response = openai_client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    return response.choices[0].message.content

# ---------------- GROQ ---------------- #

def ask_groq(prompt, model_name=DEFAULT_GROQ_MODEL):

    response = groq_client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    return response.choices[0].message.content
