import os
from config import Config
from dotenv import load_dotenv

def init():
    load_dotenv()
    config = Config()
    return config

def get_prompt(prompt_name):
    prompt_path = f"prompt/{prompt_name}.md"
    with open(prompt_path, "r", encoding="utf-8") as f:
        markdown_prompt = f.read()
    return markdown_prompt