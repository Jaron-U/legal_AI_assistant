from openai import OpenAI
from typing import List, Dict, Callable
from src.summarizer import Summarizer

class LLModel:
    def __init__(self, base_url: str, api_key: str, 
                 model_name: str, sys_prompt_func: Callable[..., str] = None,
                 stream = False, max_tokens = 1024,
                 temperature = 1, top_p = 1,
                 dialog_summary: bool = False, summarizer: Summarizer = None,
                 max_round_dialog: int = 10, min_round_dialog: int = 3,
                 obj_name: str = None, **kwargs):
        # instance of OpenAI
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        self.obj_name = obj_name

        # request params
        self.model_name = model_name
        self.stream = stream
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        
        # dialog
        self.summarizer = summarizer
        self.sys_prompt_func = sys_prompt_func
        self.sys_prompt = []
        self.messages = []
        self.max_round_dialog = max_round_dialog
        self.min_round_dialog = min_round_dialog
        self.dialog_history = []
        self.dialog_summary = dialog_summary

        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def temp_add_user_message(self, user_input:str):
        user_message = {"role":"user", "content":user_input}
        self.dialog_history.append(user_message)
    
    def remove_last_message(self):
        self.dialog_history.pop()

    def add_user_message(self, user_input:str):
        self.temp_add_user_message(user_input)
        self._summarized_or_trim()
    
    def add_assistant_message(self, assistant_output:str):
        assistant_message = {"role":"assistant", "content":assistant_output}
        self.dialog_history.append(assistant_message)
        self._summarized_or_trim()

    def _summarized_or_trim(self):
        if self.max_round_dialog == 0:
            self.dialog_history = []
            return

        current_round = len(self.dialog_history) // 2
        if current_round > self.max_round_dialog:
            if self.dialog_summary and self.summarizer:
                num_rounds_to_trim = current_round - self.min_round_dialog
                idx = num_rounds_to_trim * 2
                earliest_round = self.dialog_history[:idx]
                summary = self.summarizer.get_summary(earliest_round)
                self.dialog_history = self.dialog_history[idx:]
                self.dialog_history.insert(
                    0, {"role": "assistant", "content": f"历史对话总结: {summary}"}
                )
            else:
                self.dialog_history = self.dialog_history[-self.min_round_dialog*2:]
    
    def _get_response(self):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.messages,
            stream=self.stream,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p
        )
        
        return response

    def get_response(self, print_response=True) -> str:
        if self.sys_prompt_func:
            self.sys_prompt = [{"role": "system", "content": self.sys_prompt_func()}]
            self.messages = self.sys_prompt + self.dialog_history
        else:
            self.messages = self.dialog_history
        response = self._get_response()
        return self._process_response(response, print_response)

    def _process_response(self, response, print_response:bool) -> str:        
        if not self.stream:
            content = response.choices[0].message.content
            if print_response:
                print(f"Assistant: {content}")
            return content
        
        if print_response:
            print("\nAssistant: ", end="")
            full_content = []
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True)
                full_content.append(content)
            print()
        else:
            full_content = []
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                full_content.append(content)
        return "".join(full_content)
    
    def print_messages(self):
        if self.sys_prompt_func:
            self.messages = self.sys_prompt + self.dialog_history
        print(f"---------------{self.obj_name}----------------")
        for message in self.messages:
            print(message)
        print(f"-------------------------------")




        

