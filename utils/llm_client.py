import requests
from config.settings import SETTINGS
import json
class LLMClient:
    """
    Client for interacting with a language model (LLM) API endpoint.
    Loads configuration from SETTINGS and provides a method to generate completions.
    """
    def __init__(self):
        """
        Initialize the LLMClient with configuration from SETTINGS.
        """
        self.endpoint = SETTINGS["llm"]["endpoint"]
        self.model = SETTINGS["llm"]["model"]
        self.temperature = SETTINGS["llm"]["temperature"]
        self.max_tokens = SETTINGS["llm"]["max_tokens"]
        self.timeout = SETTINGS["llm"]["timeout"]

    def generate(self, prompt):
        """
        Generate a completion from the LLM for the given prompt.

        Args:
            prompt (str): The prompt to send to the LLM.
        Returns:
            str: The generated text from the LLM.
        Raises:
            requests.HTTPError: If the LLM API request fails.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        response = requests.post(self.endpoint, json=payload, timeout=self.timeout)
        response.raise_for_status()
        lines = response.text.strip().splitlines()
        text_chunks = []
        for line in lines:
            try:
                obj = json.loads(line)
                if "response" in obj:
                    text_chunks.append(obj["response"])
            except Exception:
                continue
        return "".join(text_chunks).strip() 