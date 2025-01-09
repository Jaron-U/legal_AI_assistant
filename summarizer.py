from config import Config
from utils import get_sys_prompt
from openai import OpenAI
from typing import List, Dict

class Summarizer:
    def __init__(self, config: Config, model_name = "qwen/qwen-2.5-7b-instruct",  
                 stream = False, max_tokens = 32000, temperature = 1, top_p = 1,
                 presence_penalty = 0, frequency_penalty = 0, response_format = { "type": "text" },
                 top_k = 50, repetition_penalty = 1, min_p = 0, system_prompt_name = "conversation_summary"):
        self.config = config
        self.model_name = model_name
        self.stream = stream
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.response_format = response_format
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.min_p = min_p
        self.repetition_penalty = repetition_penalty

        self.system_prompt_name = system_prompt_name
        self.system_prompt = get_sys_prompt(system_prompt_name)

        self.client = OpenAI(
            base_url=config.llm_api_url,
            api_key=config.llm_api_key,
        )

    def get_summary(self, conversation_buffer: List[Dict[str, str]]):
        messages = [self.system_prompt] + conversation_buffer
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=self.stream
        )
        
        return response.choices[0].message.content
        