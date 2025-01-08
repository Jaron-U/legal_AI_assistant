from openai import OpenAI
from config import Config

class LLModel:
    def __init__(self, config: Config, model_name = "qwen/qwen-2.5-72b-instruct",  
                 stream = True, max_tokens = 32000, temperature = 1, top_p = 1,
                 presence_penalty = 0, frequency_penalty = 0, response_format = { "type": "text" },
                 top_k = 50, repetition_penalty = 1, min_p = 0, system_prompt_name = "legal_assistant"):
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
        self.system_prompt = _get_sys_prompt(system_prompt_name)

        self.prompt_history = [] # include system prompt and conversation history

        self.client = OpenAI(
            base_url=config.llm_api_url,
            api_key=config.llm_api_key,
        )
        self.llm = self.client.chat.completions

        self.response = None

    def _get_response(self, conversation_history: list = None):
        messages = [self.system_prompt] + conversation_history
        self.prompt_history = messages 
        response = self.llm.create(
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

    def get_response(self, conversation_history: list = None, print_response = False):
        _ = self._get_response(conversation_history)

        if print_response:
            response = self._print_response()
        else:
            response = self._split_response()
            
        return response

    def _split_response(self, i_response=None):
        if i_response:
            response = i_response
        else:
            response = self.response

        if self.stream:
            full_content = ""
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                full_content += content
        else:
            full_content = response.choices[0].message.content
        
        return full_content
    
    def _print_response(self, i_response=None):
        if i_response:
            response = i_response
        else:
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

    def print_prompt_history(self, i_conversation_history=None):
        if i_conversation_history:
            conversation_history = i_conversation_history
        else:
            conversation_history = self.prompt_history

        print("--------------history------------------")
        for message in conversation_history:
            print(f"\n{message['role']}: {message['content']}")
        print("----------------end----------------")

def _get_sys_prompt(prompt_name: str):
    prompt_path = f"prompt/{prompt_name}.md"
    with open(prompt_path, "r", encoding="utf-8") as f:
        markdown_prompt = f.read()
    return {"role": "system", "content": markdown_prompt}