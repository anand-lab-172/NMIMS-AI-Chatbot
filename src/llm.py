import os
import ollama

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

# ---------------- LOCAL LLM ---------------- #

def ask_local_llm(prompt, model_name):

    response = ollama.chat(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]

# ---------------- GEMINI ---------------- #

def ask_gemini(prompt, model_name):

    response = gemini_client.models.generate_content(
        model=model_name,
        contents=prompt
    )

    return response.text

# ---------------- CHATGPT ---------------- #

def ask_chatgpt(prompt, model_name):

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

def ask_groq(prompt, model_name):

    response = groq_client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content
