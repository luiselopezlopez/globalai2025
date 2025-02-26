from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.messages import AgentEvent, ChatMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_ext.agents.web_surfer import MultimodalWebSurfer
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
import asyncio

from dotenv import load_dotenv
import os

from tool_sample import elmashermoso

load_dotenv()

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
    system_message="Eres asistente virtual generico. Eres un apasionado de la historia. Te gusta proporcionar información útil y relevante y muchos datos sobre cualquier tema que estes hablando.",
)

web_surfer_agent = MultimodalWebSurfer(
    name="MultimodalWebSurfer",
    model_client=model_client,
)

team = RoundRobinGroupChat([web_surfer_agent], max_turns=1)
task = "Presentate adecuadamente."

while True:
    loop= asyncio.get_event_loop()
    loop.run_until_complete(Console(team.run_stream(task=task)))
    task = input ("Por favor, escriba lo que desea (escribe 'exit' para salir):")
    if task == "exit":
        break
