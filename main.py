import json
import os
import re
from dotenv import load_dotenv
from prompt_test import call_model

from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context

load_dotenv()

langfuse = Langfuse()

issue_path = 'generation_issues\issue_17.json'
issue_id = re.search(r'issue_\d+', issue_path).group()
system_prompt_version = 10
action_response_format_version = 3
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
    """Enhanced version that handles both prompt formats"""
    def clean_content(text):
        # Normalize whitespace and newlines
        text = normalize_string(text)
        # Remove the dynamic parts that will be different between template and actual content
        text = re.sub(r'USEFUL_PROGRAMS:.*?Basic API Functions:', 
                     'USEFUL_PROGRAMS: {programs} Basic API Functions:', 
                     text)
        text = re.sub(r'Here are some useful programs.*?At each round of conversation', 
                     'Here are some useful programs written with Mineflayer APIs.\n{programs}\nAt each round of conversation', 
                     text)
        text = re.sub(r'RESPONSE FORMAT:.*', 
                     'RESPONSE FORMAT:\n{response_format}', 
                     text)
        return text

    cleaned_content = clean_content(content)
    
    for key, value in prompts.items():
        try:
            unescaped_value = unescape_json_string(value)
            cleaned_value = clean_content(unescaped_value)
            
            # Compare the cleaned versions
            if cleaned_content == cleaned_value:
                return key
            
            # Fallback comparison ignoring the dynamic parts completely
            stripped_content = cleaned_content.replace('{programs}', '').replace('{response_format}', '')
            stripped_value = cleaned_value.replace('{programs}', '').replace('{response_format}', '')
            if stripped_content in stripped_value:
                return key
        except Exception as e:
            print(f"Warning: Error processing prompt {key}: {str(e)}")
            continue
    
    return None

@observe()
def deformat_prompt(formatted_prompt):
    # Detect which version of the prompt we're dealing with
    is_new_version = "INFORMATION PROVIDED FOR EACH TASK:" in formatted_prompt
    
    if is_new_version:
        # New version pattern - looking for content between "USEFUL_PROGRAMS:" and "Basic API Functions:"
        programs_pattern = r'USEFUL_PROGRAMS:\s*(.*?)\s*Basic API Functions:'
        replacement_template = 'USEFUL_PROGRAMS: {programs} Basic API Functions:'
    else:
        # Old version pattern
        programs_pattern = r'Here are some useful programs written with Mineflayer APIs\.\n(.*?)At each round of conversation'
        replacement_template = 'Here are some useful programs written with Mineflayer APIs.\n{programs}\nAt each round of conversation'

    response_format_pattern = r'RESPONSE FORMAT:\n(.*)'

    programs = re.search(programs_pattern, formatted_prompt, re.DOTALL)
    response_format = re.search(response_format_pattern, formatted_prompt, re.DOTALL)

    if not programs:
        raise ValueError("Could not extract programs section from the prompt")

    deformatted_prompt = re.sub(programs_pattern, replacement_template, formatted_prompt, flags=re.DOTALL)
    deformatted_prompt = re.sub(response_format_pattern, 'RESPONSE FORMAT:\n{response_format}', deformatted_prompt, flags=re.DOTALL)

    added_data = {
        "programs": programs.group(1).strip() if programs else "",
        "response_format": langfuse.get_prompt("action_response_format.txt", version=action_response_format_version).compile()
    }

    return deformatted_prompt, added_data

@observe()
def process_model(model, api_key, system_prompt_version, system_content, user_content, prompts, added_data, temperature):
    # Determine if we need to find a matching prompt
    needs_prompt_match = any([
        "Here are some useful programs written with Mineflayer APIs." in system_content,
        "INFORMATION PROVIDED FOR EACH TASK:" in system_content
    ])
    
    if needs_prompt_match:
        system_prompt_key = find_matching_prompt(system_content, prompts)
    else:
        system_prompt_key = "action_template.txt"  # Default to action template if no specific match needed
    
    if system_prompt_key is None:
        print("\nDebug info for prompt matching:")
        print("Content start:", system_content[:200])
        print("Available keys:", list(prompts.keys()))
        raise ValueError("No matching system prompt found in prompts.json")

    print(f"\nTesting system prompt {system_prompt_key}, version {system_prompt_version} with model {model} for user prompt: ")
    print(user_content)

    response = call_model(system_prompt_key, system_prompt_version, user_content, model, temperature, api_key, added_data)

    langfuse_context.update_current_trace(
        name="process_system_prompt",
        tags=[
            f"Issue: {issue_id}",
            f"Prompt tested: {system_prompt_key}",
            f"Prompt version: {system_prompt_version}",
            f"Model: {model}",
            f"Temperature: {temperature}"
        ]
    )
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
    try:
        added_data = None
        generation_issue = load_json_file(issue_path)
        prompts = load_json_file('system_prompts/prompts.json')

        system_content = next((item['content'] for item in generation_issue if item['role'] == 'system'), None)
        if system_content is None:
            raise ValueError("No system content found in generation_issue.json")

        # Check for both old and new format markers
        needs_deformat = any([
            "Here are some useful programs written with Mineflayer APIs." in system_content,
            "INFORMATION PROVIDED FOR EACH TASK:" in system_content
        ])
        
        if needs_deformat:
            try:
                res = deformat_prompt(system_content)
                system_content = res[0]
                added_data = res[1]
            except Exception as e:
                print(f"Error deformatting prompt: {str(e)}")
                print("Original content:", system_content[:200])
                raise

        user_content = next((item['content'] for item in generation_issue if item['role'] == 'user'), None)
        if user_content is None:
            raise ValueError("No user content found in generation_issue.json")

        for model, api_key in models.items():
            process_model(model, api_key, system_prompt_version, system_content, user_content, prompts, added_data, temperature)
    
    except Exception as e:
        print(f"Error in main: {str(e)}")
        print("Current system content:", system_content[:200] + "..." if system_content else None)
        raise

if __name__ == "__main__":
    main()