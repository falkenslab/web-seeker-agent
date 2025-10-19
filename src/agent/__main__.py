import sys

from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI
from IPython.display import Image

from .agent import Agent

def main():

    tool = TavilySearch(max_results=4)

    prompt = """
        Eres un asistente de investigación inteligente. Utiliza el motor de búsqueda para buscar información. \
        Se te permite hacer múltiples llamadas (ya sea juntas o en secuencia). \
        Solo busca información cuando estés seguro de lo que quieres. \
        Si necesitas buscar alguna información antes de hacer una pregunta de seguimiento, ¡se te permite hacerlo
    """.strip()

    model = ChatOpenAI(model="gpt-5")
    agent = Agent(model, [tool], system=prompt)

    result = agent.ask(sys.argv[1] if len(sys.argv) > 1 else "¡Hola! ¿Cómo estás?")
    result.pretty_print()

if __name__ == "__main__":
    main()
