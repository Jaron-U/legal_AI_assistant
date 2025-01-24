from openai import OpenAI
from typing import Callable, Any

class Summarizer:
    def __init__(self, base_url: str, api_key: str,
                    model_name: str, prompt_func: Callable[..., str] = None,
                    stream = False, max_tokens = 512, 
                    temperature = 0.7, top_p = 0.9):
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )

        self.model_name = model_name
        self.stream = stream
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

        self.prompt_func = prompt_func

    def get_summary(self, content: Any) -> str:
        message = self.prompt_func(content)
        messages = [{"role": "user", "content": message}]
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=self.stream,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p
        )
        
        return response.choices[0].message.content
        