import sys

from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI

from .utils import bold, BLUE
from .agent import Agent

def main():

    tool = TavilySearch(max_results=4)

    prompt = """
        Eres un asistente de investigación inteligente. Utiliza el motor de búsqueda para buscar información. \
        Se te permite hacer múltiples llamadas (ya sea juntas o en secuencia). \
        Solo busca información cuando estés seguro de lo que quieres. \
        Si necesitas buscar alguna información antes de hacer una pregunta de seguimiento, se te permite hacerlo.
        Da respuestas breves, no te extiendas demasiado.
    """.strip()

    model = ChatOpenAI(model="gpt-4o")
    agent = Agent(model, [tool], system=prompt, verbose=False)

    #result = agent.ask(sys.argv[1] if len(sys.argv) > 1 else "¡Hola! ¿Cómo estás?")
    #result.pretty_print()

    while True:
        user_input = input(f"{bold('Tú:', BLUE)} ")
        if user_input.strip().lower() == "":
            break
        agent.ask(user_input)
        print("\n")

if __name__ == "__main__":
    main()
