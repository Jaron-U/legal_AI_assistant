from config import Config

def get_sys_prompt(prompt_name: str):
    prompt_path = f"prompt/{prompt_name}.md"
    with open(prompt_path, "r", encoding="utf-8") as f:
        markdown_prompt = f.read()
    return markdown_prompt