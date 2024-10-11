import os
from dotenv import load_dotenv
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context

load_dotenv()

langfuse = Langfuse()

system_prompt = langfuse.get_prompt("curriculum.txt", version=1)
system_prompt = system_prompt.compile()
print(system_prompt)