import os
from openai import OpenAI

class QuoteAgent:
    def __init__(self):
        self.client = OpenAI(
            base_url=os.getenv("LITELLM_BASE_URL", "http://localhost:4000/v1"),
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.model = os.getenv("OPENAI_CHAT_MODEL", "Gemini-2.5-flash-lite")

    def get_quote(self):
        prompt = "Share an inspiring motivational quote with the author’s name."
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.choices[0].message.content
