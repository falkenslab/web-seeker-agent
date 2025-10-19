import sys

from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI
from IPython.display import Image
from langgraph.checkpoint.memory import InMemorySaver

from .agent import Agent

def main():
    """Punto de entrada de la línea de comandos para ejecutar el agente.

    Crea la herramienta de búsqueda, configura el modelo LLM, construye el
    agente y procesa una consulta recibida por argumento o un saludo por defecto.
    """

    tool = TavilySearch(max_results=4)

    prompt = """
        Eres un asistente de investigación inteligente. Utiliza el motor de búsqueda para buscar información. \
        Se te permite hacer múltiples llamadas (ya sea juntas o en secuencia). \
        Solo busca información cuando estés seguro de lo que quieres. \
        Si necesitas buscar alguna información antes de hacer una pregunta de seguimiento, ¡se te permite hacerlo
    """.strip()

    model = ChatOpenAI(model="gpt-4o")
    agent = Agent(model, [tool], system=prompt, verbose=False)

    #result = agent.ask(sys.argv[1] if len(sys.argv) > 1 else "¡Hola! ¿Cómo estás?")
    #result.pretty_print()

    agent.ask(sys.argv[1] if len(sys.argv) > 1 else "¡Hola! ¿Cómo estás?", sync=False)

if __name__ == "__main__":
    main()
