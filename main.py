import json
import os
import re
from dotenv import load_dotenv
from prompt_test import call_model

from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context

load_dotenv()

langfuse = Langfuse()


issue_path = 'generation_issues\issue_0.json'
issue_id = re.search(r'issue_\d+', issue_path).group()
system_prompt_version = 6
action_response_format_version = 1
models = {
    #"gpt-3.5-turbo": os.getenv('OPENAI_API_KEY'),
    'gpt-4o-mini-2024-07-18': os.getenv('OPENAI_API_KEY'),
    #"gpt-4o": os.getenv('OPENAI_API_KEY'),
    #"meta-llama/llama-3.1-405b-instruct:free": os.getenv('OPENROUTER_API_KEY') 
}
temperature = 1



@observe()
def load_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

@observe()
def normalize_string(s):
    return ' '.join(s.replace('\n', ' ').split()).replace('\\"', '"')

@observe()
def unescape_json_string(s):
    return json.loads(f'"{s}"')

@observe()
def find_matching_prompt(content, prompts):
    normalized_content = normalize_string(content)
    for key, value in prompts.items():
        unescaped_value = unescape_json_string(value)
        normalized_value = normalize_string(unescaped_value)
        if normalized_content == normalized_value:
            return key
    return None

@observe()
def deformat_prompt(formatted_prompt):
    programs_pattern = r'Here are some useful programs written with Mineflayer APIs.\n(.*?)At each round of conversation'
    response_format_pattern = r'RESPONSE FORMAT:\n(.*)'

    programs = re.search(programs_pattern, formatted_prompt, re.DOTALL)
    response_format = re.search(response_format_pattern, formatted_prompt, re.DOTALL)

    deformatted_prompt = re.sub(programs_pattern, 'Here are some useful programs written with Mineflayer APIs.\n{programs}\nAt each round of conversation', formatted_prompt, flags=re.DOTALL)
    deformatted_prompt = re.sub(response_format_pattern, 'RESPONSE FORMAT:\n{response_format}', deformatted_prompt, flags=re.DOTALL)

    added_data = {
        "programs": programs.group(1).strip() if programs else "",
        "response_format": langfuse.get_prompt("action_response_format.txt", version=action_response_format_version).compile()#response_format.group(1).strip() if response_format else ""
    }

    return deformatted_prompt, added_data

@observe()
def process_model(model, api_key, system_prompt_version, system_content, user_content, prompts, added_data, temperature):
    if "Here are some useful programs written with Mineflayer APIs." not in system_content:
        system_prompt_key = find_matching_prompt(system_content, prompts)
    else:
        system_prompt_key = find_matching_prompt(system_content, prompts)
    
    if system_prompt_key is None:
        raise ValueError("No matching system prompt found in prompts.json")

    print(f"\nTesting system prompt {system_prompt_key}, version {system_prompt_version} with model {model} for user prompt: ")
    print(user_content)

    response = call_model(system_prompt_key, system_prompt_version, user_content, model, temperature, api_key, added_data)

    langfuse_context.update_current_trace(name = "process_system_prompt",tags=[f"Issue: {issue_id}",f"Prompt tested: {system_prompt_key}", f"Prompt version: {system_prompt_version}", f"Model: {model}", f"Temperature: {temperature}"])
    trace_id = langfuse_context.get_current_observation_id()

    if response:
        langfuse.create_dataset_item(
                dataset_name=system_prompt_key,
                source_trace_id=trace_id,
                input=user_content,
                expected_output=response,
                metadata={
                    "model": model,
                    "system_prompt": system_prompt_key,
                    "system_prompt_version": system_prompt_version,
                    "temperature": temperature,
                    "generation_issue": issue_path
                    }
                )
    langfuse_context.flush()

    print("\nModel response:")
    print(response)

@observe()
def main():
    added_data = None
    generation_issue = load_json_file(issue_path)
    prompts = load_json_file('system_prompts/prompts.json')

    system_content = next((item['content'] for item in generation_issue if item['role'] == 'system'), None)
    if "Here are some useful programs written with Mineflayer APIs." in system_content:
        res = deformat_prompt(system_content)
        system_content = res[0]
        added_data = res[1]

    user_content = next((item['content'] for item in generation_issue if item['role'] == 'user'), None)

    if system_content is None or user_content is None:
        raise ValueError("Missing system or user content in generation_issue.json")

    for model, api_key in models.items():
        process_model(model, api_key, system_prompt_version, system_content, user_content, prompts, added_data, temperature)

if __name__ == "__main__":
    main()