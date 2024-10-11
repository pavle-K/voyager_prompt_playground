from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context

load_dotenv()

langfuse = Langfuse()

class SimplifiedCriticAgent:
    def __init__(self, model_name="gpt-3.5-turbo", temperature=0):
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)

    def render_system_message(self):
        return SystemMessage(content=langfuse.get_prompt("critic.txt", version=1).compile())

    def render_human_message(self, observation, task, context):
        content = f"Observation: {observation}\n\nTask: {task}\n\nContext: {context}"
        return HumanMessage(content=content)

    def check_task_success(self, observation, task, context):
        messages = [
            self.render_system_message(),
            self.render_human_message(observation, task, context)
        ]
        print(messages)
        
        response = self.llm.invoke(messages).content
        print(response)
        # For simplicity, we'll assume the response is in the format: "Success: True/False\nCritique: ..."
        success = "Success: True" in response
        critique = response.split("Critique: ")[-1] if "Critique: " in response else ""
        
        return success, critique

def test_critic_agent():
    critic = SimplifiedCriticAgent()
    observation = "The player has successfully mined 3 wood logs."
    task = "Mine 3 wood logs"
    context = "The player needs wood to craft basic tools."
    
    success, critique = critic.check_task_success(observation, task, context)
    print(f"Task success: {success}")
    print(f"Critique: {critique}")

if __name__ == "__main__":
    test_critic_agent()