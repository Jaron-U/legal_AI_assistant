import os
from openai import OpenAI
from config import Config

class LLModel:
    def __init__(self, config: Config):
        self.config = config
        self.client = OpenAI(
            base_url=config.llm_api_url,
            api_key=config.llm_api_key,
        )
        self.llm = self.client.chat.completions
        self.response = None
        self.conversation_history = []
    
    def reset_conversation_history(self):
        self.conversation_history = []
    
    def add_to_conversation_history(self, role: str, content: str):
        self.conversation_history.append({"role": role, "content": content})

    def get_response(self, i_conversation_history: list = None):
        if i_conversation_history is None:
            conversation_history = self.conversation_history
        else:
            conversation_history = i_conversation_history

        response = self.llm.create(
            model=self.config.qwen72b_model,
            messages=conversation_history,
            stream=self.config.stream,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            presence_penalty=self.config.presence_penalty,
            frequency_penalty=self.config.frequency_penalty,
            response_format=self.config.response_format,
            extra_body={
                "top_k": self.config.top_k,
                "repetition_penalty": self.config.repetition_penalty,
                "min_p": self.config.min_p,
            }
        )
        self.response = response
        return response

    def print_response(self, i_response=None):
        if i_response:
            response = i_response
        else:
            response = self.response

        full_content = ""

        if self.config.stream:
            print("\nAssistant: ", end="")
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True)
                full_content += content
            print()
        else:
            assistant_response = response.choices[0].message.content
            print(f"Assistant: {assistant_response}")
        return full_content

    def print_conversation_history(self):
        print("--------------history------------------")
        for message in self.conversation_history:
            print(f"\n{message['role']}: {message['content']}")
        print("----------------end----------------")