import json
import os
from prompt_test import call_gpt_model

def load_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)
    
def normalize_string(s):
    return ' '.join(s.split())

def find_matching_prompt(content, prompts):
    normalized_content = normalize_string(content)
    for key, value in prompts.items():
        unescaped_value = json.loads(f'"{value}"')
        normalized_value = normalize_string(unescaped_value)
        if normalized_content == normalized_value:
            return key
    return None

def main():
    generation_issue = load_json_file('generation_issues/generation_issue.json')

    prompts = load_json_file('system_prompts\prompts.json')

    system_content = next((item['content'] for item in generation_issue if item['role'] == 'system'), None)
    user_content = next((item['content'] for item in generation_issue if item['role'] == 'user'), None)

    if system_content is None or user_content is None:
        raise ValueError("Missing system or user content in generation_issue.json")

    system_prompt_key = find_matching_prompt(system_content, prompts)
    if system_prompt_key is None:
        raise ValueError("No matching system prompt found in prompts.json")

    system_prompt_version = 1
    model = "gpt-3.5-turbo"

    print("Testing system prompt ", system_prompt_key, " , version ", system_prompt_version," with model ", model, " for user prompt: ")
    print(user_content)

    api_key = os.getenv('OPENAI_API_KEY')
    response = call_gpt_model(system_prompt_key, system_prompt_version, user_content, model, api_key)

    print("Model response:")
    print(response)

if __name__ == "__main__":
    main()