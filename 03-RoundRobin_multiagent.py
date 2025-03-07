from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.messages import AgentEvent, ChatMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_ext.models.azure import AzureAIChatCompletionClient

from azure.core.credentials import AzureKeyCredential
import asyncio

from dotenv import load_dotenv
import os

load_dotenv()

gpt_client = AzureOpenAIChatCompletionClient(
    azure_deployment=os.getenv("model-name"),
    model=os.getenv("model-name"),
    api_version=os.getenv("api-version"),
    azure_endpoint=os.getenv("azure_endpoint"),
    api_key=os.getenv("api_key")
)

deepseek_client = AzureAIChatCompletionClient(
    model=os.getenv("R1_model"),
    endpoint=os.getenv("R1_endpoint"),
    credential=AzureKeyCredential(os.getenv("R1_credential")),
    model_info={
        "json_output": False,
        "function_calling": False,
        "vision": False,
        "family": "unknown",
    },
)

deepseekAgent = AssistantAgent(
    "DeepseekR1Bot",
    model_client=deepseek_client,
    system_message="Eres un Capibara que escribe historias de miedo para adultos. Mantén la historia corta.",
)

openaiAgent = AssistantAgent(
    "gpt4oBot",
    model_client=gpt_client,
    system_message="Eres un capibara que proporciona comentarios constructivos sobre historias de miedo para adultos para agregar un final inesperado e impactante. Responde con 'APROBADO' cuando se aborden tus comentarios.",
)

text_termination = TextMentionTermination("APROBADO")

team = RoundRobinGroupChat(
    [deepseekAgent, openaiAgent],
    max_turns=50,
    termination_condition=text_termination
)


async def main():
    question = input("Por favor, ingrese el tema de la historia: ")
    await Console(team.run_stream(task=question))


# Run the asynchronous function
if __name__ == "__main__":
    asyncio.run(main())