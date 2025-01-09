from openai import OpenAI
from config import Config
from typing import List, Dict
from utils import get_sys_prompt
from summarizer import Summarizer

class LLModel:
    def __init__(self, config: Config, summarizer: Summarizer,
                 model_name = "qwen/qwen-2.5-72b-instruct",  
                 stream = True, max_tokens = 32000, temperature = 1, top_p = 1,
                 presence_penalty = 0, frequency_penalty = 0, response_format = { "type": "text" },
                 top_k = 50, repetition_penalty = 1, min_p = 0, system_prompt_name = "legal_assistant"):
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

        self.summarizer = summarizer
        self.conversation_buffer = []
        self.keep_k = config.conversation_buffer_keep_k

        self.client = OpenAI(
            base_url=config.llm_api_url,
            api_key=config.llm_api_key,
        )

        self.response = None
    
    def set_conversation_buffer(self, conversation_buffer: List[Dict[str, str]]):
        self.conversation_buffer = conversation_buffer
    
    def add_user_message(self, message: str):
        self._add_message("user", message)
    
    def add_assistant_message(self, message: str):
        self._add_message("assistant", message)
    
    def remove_last_message(self):
        self.conversation_buffer.pop()
    
    def print_conversation_buffer(self):
        for msg in self.conversation_buffer:
            print(msg)

    def _add_message(self, role: str, message: str) -> None:
        self.conversation_buffer.append({"role": role, "content": message})
        self._maybe_summarize_and_trim()
    
    def _maybe_summarize_and_trim(self) -> None:
        current_rounds = len(self.conversation_buffer) // 2
        while current_rounds > self.keep_k:
            earliest_round = self.conversation_buffer[:2]
            summary = self.summarizer.get_summary(earliest_round)

            self.conversation_buffer = self.conversation_buffer[2:]
            self.conversation_buffer.insert(
                0, {"role": "assistant", "content": f"历史对话总结: {summary}"}
            )

            current_rounds = len(self.conversation_buffer) // 2

    def _get_response(self):
        messages = [self.system_prompt] + self.conversation_buffer
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=self.stream,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            response_format=self.response_format,
            extra_body={
                "top_k": self.top_k,
                "repetition_penalty": self.repetition_penalty,
                "min_p": self.min_p,
            }
        )
        
        self.response = response
        return response

    def get_response(self, print_response = False):
        _ = self._get_response()

        if print_response:
            response = self._print_response()
        else:
            response = self._split_response()
            
        return response

    def _split_response(self, response=None):
        if response is None:
            response = self.response

        if self.stream:
            full_content = ""
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                full_content += content
        else:
            full_content = response.choices[0].message.content
        
        return full_content
    
    def _print_response(self, response=None):
        if response is None:
            response = self.response
        
        full_content = ""

        if self.stream:
            print("\nAssistant: ", end="")
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True)
                full_content += content
            print()
        else:
            full_content = response.choices[0].message.content
            print(f"Assistant: {full_content}")
        return full_content

    def print_prompt_history(self, conversation_history=None):
        if conversation_history is None:
            conversation_history = [self.system_prompt] + self.conversation_buffer
        
        print(f"--------------{self.system_prompt_name}-----------------")
        for message in conversation_history:
            print(f"\n{message['role']}: {message['content']}")
        print("----------------end----------------")
