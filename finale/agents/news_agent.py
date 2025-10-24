import os
from openai import OpenAI

class NewsAgent:
    def __init__(self):
        self.client = OpenAI(
            base_url=os.getenv("LITELLM_BASE_URL", "http://localhost:4000/v1"),
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.model = os.getenv("OPENAI_CHAT_MODEL", "Gemini-2.5-flash-lite")

    def get_news(self):
        prompt = "Give me 3 trending tech news headlines with one line summary each."
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.choices[0].message.content
