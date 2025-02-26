from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.messages import AgentEvent, ChatMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
import asyncio

from dotenv import load_dotenv
import os

from tool_sample import elmashermoso

load_dotenv()

async def main() -> None:
    model_client = AzureOpenAIChatCompletionClient(
        azure_deployment=os.getenv("model-name"),
        model=os.getenv("model-name"),
        api_version=os.getenv("api-version"),
        azure_endpoint=os.getenv("azure_endpoint"),
        api_key=os.getenv("api_key")
    )

    assistantAgent = AssistantAgent(
        "assistantBot",
        model_client=model_client,
        tools=[elmashermoso],
    )

    termination = TextMentionTermination("TERMINATE")
    team = RoundRobinGroupChat([assistantAgent], termination_condition=termination)
    question = input("Por favor, ingrese su pregunta: ")
    await Console(team.run_stream(task=question))

while True:
    asyncio.run(main())
