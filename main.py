from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import json
import os

app = FastAPI()

class Request(BaseModel):
    situation: str

def get_guidance(situation):
    client = OpenAI(
    api_key=os.environ.get("GROQ_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": """You are a Christian assistant.

    Return ONLY valid JSON. No extra text.

    Format:
    {
    "issue": "a natural, human-readable description of the emotional/spiritual issue",
    "verses": [
        {
        "reference": "Bible verse reference",
        "text": "exact verse text",
        "explanation": "how it applies to the situation"
        }
    ]
    }

    Rules:
    - Use 2–4 real Bible verses
    - Be accurate (no made-up verses)
    - Keep explanations clear and practical
    - DO NOT include any text outside the JSON
    """
            },
            {
                "role": "user",
                "content": "I'm feeling anxious about my future"
            }
        ]
    )

    return response.choices[0].message.content

@app.post("/get-guidance")
def get_guidance_endpoint(request: Request):
    result = get_guidance(request.situation)
    return json.loads(result)

