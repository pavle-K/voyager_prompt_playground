from openai import OpenAI
from dotenv import load_dotenv
from langfuse import Langfuse
import requests
import json

load_dotenv()

langfuse = Langfuse()

def call_gpt_model(system_prompt_key, system_prompt_version, user_prompt, model, api_key, added_data):

    system_prompt = langfuse.get_prompt(system_prompt_key, version=system_prompt_version).compile()

    if added_data:
        system_prompt = system_prompt.format(**added_data)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    if 'gpt' in model:
        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model=model,
            messages=messages
        )

    
    elif 'llama' in model:
        client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
        )

        response = client.chat.completions.create(
                model=model,
                messages=messages
            )
        
        """
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
            },
            data=json.dumps({
                "model": model, # Optional
                "messages": messages
            })
        )
        print(response)
        """

    return response.choices[0].message.content