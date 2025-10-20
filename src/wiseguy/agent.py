import asyncio
import operator
from typing import TypedDict, Annotated, Iterator

from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver

from .utils import bold, PURPLE

class AgentState(TypedDict):
    """Estado del agente usado por el grafo.

    Contiene la lista de mensajes intercambiados durante la conversación.

    - messages: lista acumulativa de mensajes; se combina usando ``operator.add``.
    """

    messages: Annotated[list[AnyMessage], operator.add]  # Crea una lista de mensajes que se puede concatenar con el operador +


class Agent:
    """Agente orquestador que combina un modelo LLM con herramientas.

    Construye un grafo de estados con dos nodos principales:
    - "llm": invoca el modelo para generar la siguiente acción o respuesta.
    - "action": ejecuta herramientas solicitadas por el modelo y retorna sus resultados.

    El grafo itera hasta que el modelo no solicite más herramientas.

    Atributos:
        system: texto del mensaje de sistema que se añadirá una vez al principio.
        graph: grafo compilado de LangGraph que gestiona el flujo de mensajes.
        tools: diccionario de herramientas disponibles indexadas por nombre.
        model: modelo LLM con herramientas enlazadas mediante ``bind_tools``.
    """

    _current_state: AgentState = None

    def __init__(self, model, tools, system: str = "", verbose: bool = False):
        """Inicializa el agente con un modelo, herramientas y un prompt de sistema.

        Parámetros:
            model: instancia del modelo conversacional (por ejemplo, ``ChatOpenAI``).
            tools: lista de herramientas compatibles con LangChain a exponer al modelo.
            system: mensaje de sistema opcional que se antepone solo una vez.
        """
        self.system = system
        self.verbose = verbose
        self.config = {
            "recursion_limit": 50,                              # Límite de recursión para evitar bucles infinitos en el grafo (para que no esté infinitamente dando vueltas)
            "configurable": { "thread_id" : "1"}                # Identificador del hilo de conversación (en este caso, 1 agente sólo puede mantener una conversación a la vez)
        }

        graph = StateGraph(AgentState)

        # Definición de nodos del grafo
        graph.add_node("init", self.init_state)                 # Nodo de estado inicial
        graph.add_node("llm", self.call_openai)                 # Nodo de llamada al modelo
        graph.add_node("action", self.take_action)              # Nodo de ejecución de herramientas (acciones)

        # Definición de aristas del grafo
        graph.add_edge("init", "llm")                           # Desde el estado inicial, ir al modelo        
        graph.add_conditional_edges(
            "llm",                                              # La arista condicional sale del nodo "llm"
            self.exists_action,                                 # Función que decide si se debe ir al nodo de acción o terminar
            {True: "action", False: END},
        )                                                       # Si el modelo decide llamar a una herramienta, ir al nodo de acción; si no, terminar
        graph.add_edge("action", "llm")                         # Después de ejecutar una acción, volver al modelo

        # Definición del punto de entrada del grafo
        graph.set_entry_point("init")

        # Compilación del grafo para su ejecución
        self.graph = graph.compile(checkpointer=InMemorySaver())

        # Mapeo de herramientas por nombre
        self.tools = {t.name: t for t in tools}

        # Asociar las herramientas al modelo
        self.model = model.bind_tools(tools)

    def init_state(self, state: AgentState) -> AgentState:
        """Crea el estado inicial del agente con el mensaje de sistema.

        Devuelve:
            Un estado inicial con el mensaje de sistema si está configurado.
        """
        self._current_state = {"messages": [SystemMessage(content=self.system) if self.system else []]}
        return self._current_state
    
    def exists_action(self, state: AgentState) -> bool:
        """Indica si el último mensaje contiene llamadas a herramientas.

        Parámetros:
            state: estado actual con el historial de mensajes.

        Devuelve:
            ``True`` si el último mensaje del modelo incluye ``tool_calls``; en caso contrario ``False``.
        """
        self._current_state = state
        result = state["messages"][-1]          # Último mensaje generado por el modelo
        return len(result.tool_calls) > 0       # Devuelve True si hay llamadas a herramientas en el último mensaje

    def call_openai(self, state: AgentState) -> AgentState:
        """Invoca el modelo LLM con el historial de mensajes.

        Inserta el mensaje de sistema una única vez si está configurado.

        Parámetros:
            state: estado actual con los mensajes acumulados.

        Devuelve:
            Un nuevo estado con el mensaje de salida del modelo en ``messages``.
        """
        self._current_state = state
        messages = state["messages"]
        message = self.model.invoke(messages)
        if self.verbose:
            message.pretty_print()
        return {"messages": [message]}

    def take_action(self, state: AgentState) -> AgentState:
        """Ejecuta las herramientas solicitadas por el modelo y retorna sus resultados.

        Parámetros:
            state: estado actual cuyo último mensaje contiene ``tool_calls``.

        Devuelve:
            Un estado con los mensajes de tipo ``ToolMessage`` correspondientes a cada ejecución.
        """
        self._current_state = state
        tool_calls = state["messages"][-1].tool_calls
        results = []
        for t in tool_calls:
            if self.verbose:
                print(f"\nEjecutando acción: {t}\n")
            if not t["name"] in self.tools:  # check for bad tool name from LLM
                if self.verbose:
                    print("\n ....nombre de tool no válida....")
                result = "nombre de tool no válida, reintentar"  # instruir al LLM a reintentar si el nombre es incorrecto
            else:
                result = self.tools[t["name"]].invoke(t["args"])
            message = ToolMessage(tool_call_id=t["id"], name=t["name"], content=str(result))
            if self.verbose:
                message.pretty_print()
            results.append(message)
        return {"messages": results}

    def ask(self, question: str, sync: bool = False) -> AnyMessage|None:
        """Formula una pregunta al agente y devuelve la última respuesta.

        Parámetros:
            question: texto de la consulta del usuario.

        Devuelve:
            El último mensaje generado por el agente en respuesta a la consulta o ``None`` si no hay respuesta o es asíncrono.
        """
        message = HumanMessage(content=question)
        if self.verbose:
            message.pretty_print()
        messages = [message]
        if sync:
            return self.__invoke(messages)
        else:
            return asyncio.run(self.__ainvoke(messages))

    def __invoke(self, messages: list[AnyMessage]) -> AnyMessage:
        """Formula una pregunta al agente y devuelve la última respuesta.

        Parámetros:
            question: texto de la consulta del usuario.
        """
        return self.graph.invoke(input={"messages": messages}, config=self.config)

    async def __ainvoke(self, messages: list[AnyMessage]) -> None:
        """Formula una pregunta al agente y devuelve la última respuesta.

        Parámetros:
            question: texto de la consulta del usuario.
        """
        print(f"\n{bold('🤖 Wiseguy:', PURPLE)} ", end="")
        async for event in self.graph.astream_events(input={"messages": messages}, config=self.config):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                # Si el contenido está vacío significa que el modelo está pidiendo una herramienta, por eso sólo imprimimos contenido no vacío
                if content:
                    print(content, end="")

    def print_graph(self) -> None:
        """Imprime una representación del grafo del agente."""
        print(self.graph.get_graph().draw_mermaid())