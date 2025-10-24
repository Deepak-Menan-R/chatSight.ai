import os
import json
from openai import OpenAI
from agents.news_agent import NewsAgent
from agents.weather_agent import WeatherAgent
from agents.quote_agent import QuoteAgent

class MasterAgent:
    def __init__(self):
        self.news_agent = NewsAgent()
        self.weather_agent = WeatherAgent()
        self.quote_agent = QuoteAgent()
        self.client = OpenAI(
            base_url=os.getenv("LITELLM_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.model = os.getenv("OPENAI_CHAT_MODEL", "Gemini-2.5-flash-lite")

    def handle_task(self, user_input: str):
        # Step 1: Detect all relevant agents
        routing_prompt = f"""
        You are an AI dispatcher. The user query may have multiple requests.
        Choose all relevant agents from [NewsAgent, WeatherAgent, QuoteAgent].
        Return a JSON array of agent names.
        User query: "{user_input}"
        """
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": routing_prompt}]
        )
        agent_names = resp.choices[0].message.content.strip()
        print("These are the agents to call:", agent_names)

        # Clean up the string to ensure valid JSON
        agent_names = agent_names.replace("```json", "").replace("```", "").strip()
        
        try:
            import json
            agents_to_call = json.loads(agent_names)
            # Ensure agents_to_call is a clean list of strings
            agents_to_call = [agent.strip() for agent in agents_to_call]
            print("Parsed agents to call:", agents_to_call)
        except Exception as e:
            print("JSON parse error:", e)
            return "Error parsing agent names. Please ensure the response is a valid JSON array."

        # Now agents_to_call is a clean list: ['QuoteAgent', 'WeatherAgent']
        # Step 2: Call each agent and collect responses
        final_output = []
        for agent in agents_to_call:
            agent = agent.strip()
            if agent == "NewsAgent":
                answer = self.news_agent.get_news()
            elif agent == "WeatherAgent":
                answer = self.weather_agent.get_weather()
            elif agent == "QuoteAgent":
                answer = self.quote_agent.get_quote()
            else:
                answer = f"No agent found for {agent}"
            final_output.append({"agent": agent, "answer": answer})

        return final_output
