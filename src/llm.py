import os

from dotenv import load_dotenv

from google import genai
from openai import OpenAI
from groq import Groq

load_dotenv()

# ---------------- GOOGLE ---------------- #

google_api_key = os.getenv("GOOGLE_API_KEY")

gemini_client = None

if google_api_key:

    gemini_client = genai.Client(
        api_key=google_api_key
    )

# ---------------- OPENAI ---------------- #

openai_api_key = os.getenv("OPENAI_API_KEY")

openai_client = None

if openai_api_key:

    openai_client = OpenAI(
        api_key=openai_api_key
    )

# ---------------- GROQ ---------------- #

groq_api_key = os.getenv("GROQ_API_KEY")

groq_client = None

if groq_api_key:

    groq_client = Groq(
        api_key=groq_api_key
    )

# ---------------- GEMINI ---------------- #

def ask_gemini(
    prompt,
    model_name="gemini-1.5-flash"
):

    if gemini_client is None:
        return "❌ Gemini API Key Missing"

    response = gemini_client.models.generate_content(
        model=model_name,
        contents=prompt
    )

    return response.text

# ---------------- CHATGPT ---------------- #

def ask_chatgpt(
    prompt,
    model_name="gpt-4o-mini"
):

    if openai_client is None:
        return "❌ OpenAI API Key Missing"

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

def ask_groq(
    prompt,
    model_name="llama-3.1-8b-instant"
):

    if groq_client is None:
        return "❌ Groq API Key Missing"

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
