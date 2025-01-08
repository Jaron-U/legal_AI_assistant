from config import Config
from typing import List, Dict
from llmodel import LLModel

class ConversationBuffer:
    def __init__(self, conversation_summary_model: LLModel, config: Config, 
                 conversation_buffer: List[Dict[str, str]] = []) -> None:
        self.config = config
        self.conversation_buffer: List[Dict[str, str]] = conversation_buffer
        self.keep_k = config.conversation_buffer_keep_k
        self.conversation_summary_model = conversation_summary_model
    
    def add_user_message(self, message: str) -> None:
        self._add_message("user", message)
    
    def add_assistant_message(self, message: str) -> None:
        self._add_message("assistant", message)
    
    def temp_add_user_message(self, message: str) -> None:
        self.conversation_buffer.append({"role": "user", "content": message})
    
    def remove_last_message(self) -> None:
        if self.conversation_buffer:
            self.conversation_buffer.pop()
    
    def _add_message(self, role: str, message: str) -> None:
        self.conversation_buffer.append({"role": role, "content": message})
        self._maybe_summarize_and_trim()
    
    def _conversation_summary(self, messages: List[Dict[str, str]]) -> str:
        response = self.conversation_summary_model.get_response(messages)
        return response
    
    def _maybe_summarize_and_trim(self) -> None:
        current_rounds = len(self.conversation_buffer) // 2
        while current_rounds > self.keep_k:
            earliest_round = self.conversation_buffer[:2]
            summary = self._conversation_summary(earliest_round)

            self.conversation_buffer = self.conversation_buffer[2:]
            self.conversation_buffer.insert(
                0, {"role": "assistant", "content": f"历史对话总结: {summary}"}
            )

            current_rounds = len(self.conversation_buffer) // 2
    
    def get_conversation_buffer(self) -> List[Dict[str, str]]:
        return self.conversation_buffer

    def print_conversation_buffer(self) -> None:
        for msg in self.conversation_buffer:
            print(msg)