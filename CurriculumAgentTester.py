import os
from dotenv import load_dotenv
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

langfuse = Langfuse()

class CurriculumAgentTester:
    def __init__(self, model_name="gpt-3.5-turbo", temperature=0):
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        self.qa_llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        self.prompts = {
            "curriculum": "",
            "curriculum_task_decomposition": "",
            "curriculum_qa_step1_ask_questions": "",
            "curriculum_qa_step2_answer_questions": ""
        }
        self.completed_tasks = []
        self.failed_tasks = []

    def load_prompts(self, prompt_dict):
        self.prompts.update(prompt_dict)

    @observe()
    def test_main_curriculum(self, game_state):
        messages = [
            SystemMessage(content=self.prompts["curriculum"]),
            HumanMessage(content=self.render_observation(game_state))
        ]
        response = self.llm.invoke(messages).content
        return response

    @observe()
    def test_task_decomposition(self, game_state, task):
        messages = [
            SystemMessage(content=self.prompts["curriculum_task_decomposition"]),
            HumanMessage(content=f"Game State:\n{self.render_observation(game_state)}\n\nFinal task: {task}")
        ]
        response = self.llm.invoke(messages).content
        return response

    @observe()
    def test_qa_step1(self, game_state):
        messages = [
            SystemMessage(content=self.prompts["curriculum_qa_step1_ask_questions"]),
            HumanMessage(content=self.render_observation(game_state))
        ]
        response = self.qa_llm.invoke(messages).content
        return response

    @observe()
    def test_qa_step2(self, question):
        messages = [
            SystemMessage(content=self.prompts["curriculum_qa_step2_answer_questions"]),
            HumanMessage(content=f"Question: {question}")
        ]
        response = self.qa_llm.invoke(messages).content
        return response

    @observe()
    def test_get_task_context(self, task):
        question = f"How to {task.replace('_', ' ').lower()} in Minecraft?"
        return self.test_qa_step2(question)

    def render_observation(self, game_state):
        observation = game_state.copy() if isinstance(game_state, dict) else {"raw_state": game_state}
        observation["completed_tasks"] = ", ".join(self.completed_tasks) if self.completed_tasks else "None"
        observation["failed_tasks"] = ", ".join(self.failed_tasks) if self.failed_tasks else "None"
        return "\n".join(f"{k}: {v}" for k, v in observation.items())

    def update_task_status(self, task, success):
        if success:
            self.completed_tasks.append(task)
        else:
            self.failed_tasks.append(task)

def main():
    # Ensure OpenAI API key is set
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = input("Please enter your OpenAI API key: ")

    # Initialize the tester
    tester = CurriculumAgentTester()

    # Load prompts using Langfuse
    prompts = {
        "curriculum": langfuse.get_prompt("curriculum.txt", version=1).compile(),
        "curriculum_task_decomposition": langfuse.get_prompt("curriculum_task_decomposition.txt", version=1).compile(),
        "curriculum_qa_step1_ask_questions": langfuse.get_prompt("curriculum_qa_step1_ask_questions.txt", version=1).compile(),
        "curriculum_qa_step2_answer_questions": langfuse.get_prompt("curriculum_qa_step2_answer_questions.txt", version=1).compile()
    }
    tester.load_prompts(prompts)

    game_state = {
        "biome": "Forest",
        "time": "Day",
        "nearby_blocks": "Grass, Dirt, Oak Log",
        "inventory": "Empty"
    }

    print("Testing main curriculum:")
    next_task = tester.test_main_curriculum(game_state)
    print(next_task)

    print("\nTesting task decomposition:")
    print(tester.test_task_decomposition(game_state, next_task))

    print("\nTesting QA step 1 (generate questions):")
    questions = tester.test_qa_step1(game_state)
    print(questions)

    print("\nTesting QA step 2 (answer question):")
    question = "What can I craft with oak logs?"
    print(tester.test_qa_step2(question))

    print("\nTesting get task context:")
    print(tester.test_get_task_context(next_task))

    # Simulate task completion
    tester.update_task_status(next_task, True)

    print("\nTesting main curriculum after task completion:")
    print(tester.test_main_curriculum(game_state))

if __name__ == "__main__":
    main()