import requests
from config.settings import SETTINGS

class LLMClient:
    def __init__(self):
        self.endpoint = SETTINGS["llm"]["endpoint"]
        self.model = SETTINGS["llm"]["model"]
        self.temperature = SETTINGS["llm"]["temperature"]
        self.max_tokens = SETTINGS["llm"]["max_tokens"]
        self.timeout = SETTINGS["llm"]["timeout"]

    def generate(self, prompt, temperature=None, max_tokens=None, stop=None):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens
        }
        if stop is not None:
            payload["stop"] = stop
        response = requests.post(self.endpoint, json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.text 