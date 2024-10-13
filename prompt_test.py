from openai import OpenAI
from dotenv import load_dotenv
from langfuse import Langfuse

load_dotenv()

langfuse = Langfuse()

def call_gpt_model(system_prompt_key, system_prompt_version, user_prompt, model, api_key):
    client = OpenAI(api_key=api_key)

    system_prompt = langfuse.get_prompt(system_prompt_key, version=system_prompt_version).compile()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages
    )

    return response.choices[0].message.content