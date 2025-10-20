import sys

from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI

from .utils import PURPLE, bold, BLUE
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

    # Imprime el grafo si se pasa el argumento --print-graph en formato Mermaid
    if len(sys.argv) > 1:
        arg = sys.argv[1].strip()
        if arg == "--print-graph":
            agent.print_graph()
            return

    print(f"\n{bold("🤖 Wiseguy:", PURPLE)} ¡Hola! Soy Wiseguy, tu asistente de investigación. ¿En qué puedo ayudarte hoy?\n")
    while True:
        user_input = input(f"{bold('🧑‍🦲 Tú:', BLUE)} ")
        if user_input.strip().lower() == "":
            break
        agent.ask(user_input)
        print("\n")

if __name__ == "__main__":
    main()
