from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.messages import AgentEvent, ChatMessage
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
import asyncio


from dotenv import load_dotenv
import os

load_dotenv()

# Define the Azure OpenAI Chat Completion Client
az_model_client = AzureOpenAIChatCompletionClient(
        azure_deployment=os.getenv("model-name"),
        model=os.getenv("model-name"),
        api_version=os.getenv("api-version"),
        azure_endpoint=os.getenv("azure_endpoint"),
        api_key=os.getenv("api_key")
    )

# Define a function to save the story to a file.
async def savestory(story:str, filename:str):
    with open(filename,
            'w', encoding='utf-8') as f:
        f.write(story)

# Create the Planning agent.
planning_agent = AssistantAgent(
    "PlanningAgent",
    description="Eres un agente de planificación de tareas y debes ser el primero en intervenir cuando se te asigne una nueva tarea.",
    model_client=az_model_client,
    system_message="""
        Eres un agente de planificación.
        Tu trabajo es descomponer tareas complejas en subtareas más pequeñas y manejables.

        Tus miembros del equipo son:

        Story_writer: Escribe la historia y realiza correcciones.
        Story_reviewer: Verifica si la historia es adecuada para niños y proporciona comentarios constructivos para agregar un final positivo e impactante. No escribe la historia, solo brinda comentarios y mejoras.
        Story_moral: Agrega la moraleja a la historia, cuando Story_writer y Story_reviewer hayan decidido que la historia esta completa.
        Story_editor: Una vez que la historia esté completa y la moraleja completa, convierte la historia final en un archivo.

        Tú solo planificas y delegas tareas; no las ejecutas tú mismo. Puedes involucrar a los miembros del equipo varias veces para garantizar que se proporcione una historia perfecta.

        Al asignar tareas, usa este formato:

        <agente> : <tarea>
        Cuando todas las tareas estén completas, finaliza con "TERMINATE".
    """,
)

# Create the Writer agent.
Story_writer = AssistantAgent(
    "Story_writer",
    model_client=az_model_client,
    system_message="Eres capibara que escribe historias de estilo cyber-punk. Mantén la historia corta.",
)

# Create the Reviewer agent.
Story_reviewer = AssistantAgent(
    "Story_reviewer",
    model_client=az_model_client,
    system_message="Eres un capibara que verifica si la historia es adecuada para niños y proporciona comentarios constructivos para que las historias infantiles tengan un final positivo e impactante.",
)

# Story Moral Agent.
Story_moral = AssistantAgent(
    "Story_moral",
    model_client=az_model_client,
    system_message="Eres un asistente de inteligencia artificial útil que agrega la moraleja al final de la historia para que los niños tengan un impacto positivo y un gran aprendizaje. La moraleja debe tener solo 2-3 líneas y debe escribirse con la siguiente separación: '======== Moraleja de la historia ========='",
)

# Story Editor Agent.
Story_editor = AssistantAgent(
    "Story_editor",
    model_client=az_model_client,
    tools=[savestory],
    system_message="Eres un asistente de inteligencia artificial útil que convierte la historia final en un archivo .txt",
)

text_mention_termination = TextMentionTermination("TERMINATE")
max_messages_termination = MaxMessageTermination(max_messages=15)
termination = text_mention_termination | max_messages_termination

team = SelectorGroupChat(
    [planning_agent, Story_writer, Story_reviewer, Story_moral, Story_editor],
    model_client=az_model_client,
    termination_condition=termination,
)

# Define the main asynchronous function
async def main():
    question = input("Por favor, ingrese el tema de la historia: ")
    await Console(
        team.run_stream(task=question)
    )  # Stream the messages to the console.


# Run the asynchronous function
if __name__ == "__main__":
    asyncio.run(main())