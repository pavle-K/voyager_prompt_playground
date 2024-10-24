from openai import OpenAI
from dotenv import load_dotenv
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context
import requests
import json

load_dotenv()

langfuse = Langfuse()

@observe(as_type="generation")
def call_model(system_prompt_key, system_prompt_version, user_prompt, model, temperature, api_key, added_data):
    try:
        system_prompt = langfuse.get_prompt(system_prompt_key, version=system_prompt_version).compile()
        
        if added_data:
            # Replace the format placeholders with temporary markers
            system_prompt = system_prompt.replace("{programs}", "%%PROGRAMS%%")
            system_prompt = system_prompt.replace("{response_format}", "%%RESPONSE_FORMAT%%")
            
            # Now do the substitution without format()
            system_prompt = system_prompt.replace("%%PROGRAMS%%", added_data['programs'])
            system_prompt = system_prompt.replace("%%RESPONSE_FORMAT%%", added_data['response_format'])
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        if 'gpt' in model:
            client = OpenAI(api_key=api_key)

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature= temperature
            )

    
        elif 'llama' in model:
            client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
            )

            response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature= temperature
                )

        return response.choices[0].message.content
    except Exception as e:
        print(f"Error while generating for model {model} :", e)
        return None