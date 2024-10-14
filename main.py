import json
import os
import re
from prompt_test import call_gpt_model


system_prompt_version = 1
model = 'meta-llama/llama-3.1-405b-instruct:free'#"gpt-3.5-turbo" # 'gpt-4o' 'llama3.1-405b'
if 'gpt' in model:
    api_key = os.getenv('OPENAI_API_KEY')
if 'llama' in model:
    api_key = os.getenv('OPENROUTER_API_KEY')

models = {
    "gpt-3.5-turbo": os.getenv('OPENAI_API_KEY'),
    "gpt-4o": os.getenv('OPENAI_API_KEY'),
    "meta-llama/llama-3.1-405b-instruct:free": os.getenv('OPENROUTER_API_KEY') 
}

def load_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def normalize_string(s):
    return ' '.join(s.replace('\n', ' ').split()).replace('\\"', '"')

def unescape_json_string(s):
    return json.loads(f'"{s}"')

def find_matching_prompt(content, prompts):
    normalized_content = normalize_string(content)
    for key, value in prompts.items():
        unescaped_value = unescape_json_string(value)
        normalized_value = normalize_string(unescaped_value)
        if normalized_content == normalized_value:
            return key
    return None

def deformat_prompt(formatted_prompt):
    programs_pattern = r'Here are some useful programs written with Mineflayer APIs.\n(.*?)At each round of conversation'
    response_format_pattern = r'RESPONSE FORMAT:\n(.*)'

    programs = re.search(programs_pattern, formatted_prompt, re.DOTALL)
    response_format = re.search(response_format_pattern, formatted_prompt, re.DOTALL)

    deformatted_prompt = re.sub(programs_pattern, 'Here are some useful programs written with Mineflayer APIs.\n{programs}\nAt each round of conversation', formatted_prompt, flags=re.DOTALL)
    deformatted_prompt = re.sub(response_format_pattern, 'RESPONSE FORMAT:\n{response_format}', deformatted_prompt, flags=re.DOTALL)

    added_data = {
        "programs": programs.group(1).strip() if programs else "",
        "response_format": response_format.group(1).strip() if response_format else ""
    }

    return deformatted_prompt, added_data

def main(system_prompt_version, model, api_key):
    added_data = None
    generation_issue = load_json_file('generation_issues/generation_issue.json')
    prompts = load_json_file('system_prompts/prompts.json')

    system_content = next((item['content'] for item in generation_issue if item['role'] == 'system'), None)
    if "Here are some useful programs written with Mineflayer APIs." in system_content:
        res = deformat_prompt(system_content)
        system_content = res[0]
        added_data = res[1]

    user_content = next((item['content'] for item in generation_issue if item['role'] == 'user'), None)

    if system_content is None or user_content is None:
        raise ValueError("Missing system or user content in generation_issue.json")

    if "Here are some useful programs written with Mineflayer APIs." not in system_content:
        system_prompt_key = find_matching_prompt(system_content, prompts)
    else:
        system_prompt_key = find_matching_prompt(system_content, prompts)
    
    if system_prompt_key is None:
        raise ValueError("No matching system prompt found in prompts.json")

    print(f"Testing system prompt {system_prompt_key}, version {system_prompt_version} with model {model} for user prompt: ")
    print(user_content)

    response = call_gpt_model(system_prompt_key, system_prompt_version, user_content, model, api_key, added_data)

    print("Model response:")
    print(response)

if __name__ == "__main__":
    main(system_prompt_version, model, api_key)