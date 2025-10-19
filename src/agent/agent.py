import operator
from typing import TypedDict, Annotated

from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.errors import GraphRecursionError

import agent


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

    def __init__(self, model, tools, system: str = ""):
        """Inicializa el agente con un modelo, herramientas y un prompt de sistema.

        Parámetros:
            model: instancia del modelo conversacional (por ejemplo, ``ChatOpenAI``).
            tools: lista de herramientas compatibles con LangChain a exponer al modelo.
            system: mensaje de sistema opcional que se antepone solo una vez.
        """
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)  # Nodo de llamada al modelo
        graph.add_node("action", self.take_action)  # Nodo de ejecución de herramientas (acciones)
        graph.add_conditional_edges(
            "llm",  # La arista condicional sale del nodo del modelo
            self.exists_action,  # Función que decide si se debe ir al nodo de acción o terminar
            {True: "action", False: END},
        )  # Si el modelo decide llamar a una herramienta, ir al nodo de acción; si no, terminar
        graph.add_edge("action", "llm")  # Después de ejecutar una acción, volver al modelo
        graph.set_entry_point("llm")  # El punto de entrada del grafo es la llamada al modelo
        self.graph = graph.compile()  # Compilar el grafo para su ejecución
        self.tools = {t.name: t for t in tools}  # Mapeo de herramientas por nombre
        self.model = model.bind_tools(tools)  # Asociar las herramientas al modelo

    def exists_action(self, state: AgentState) -> bool:
        """Indica si el último mensaje contiene llamadas a herramientas.

        Parámetros:
            state: estado actual con el historial de mensajes.

        Devuelve:
            ``True`` si el último mensaje del modelo incluye ``tool_calls``; en caso contrario ``False``.
        """
        self._current_state = state
        result = state["messages"][-1]
        return len(result.tool_calls) > 0

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
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
            self.system = ""  # Solo usar el mensaje de sistema una vez
        message = self.model.invoke(messages)
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
            print(f"\nEjecutando acción: {t}\n")
            if not t["name"] in self.tools:  # check for bad tool name from LLM
                print("\n ....nombre de tool no válida....")
                result = "nombre de tool no válida, reintentar"  # instruir al LLM a reintentar si el nombre es incorrecto
            else:
                result = self.tools[t["name"]].invoke(t["args"])
            message = ToolMessage(tool_call_id=t["id"], name=t["name"], content=str(result))
            message.pretty_print()
            results.append(message)
        return {"messages": results}

    def ask(self, question: str) -> AnyMessage:
        """Formula una pregunta al agente y devuelve la última respuesta.

        Parámetros:
            question: texto de la consulta del usuario.

        Devuelve:
            El último ``AnyMessage`` producido tras ejecutar el grafo (respuesta final del flujo).
        """
        message = HumanMessage(content=question)
        message.pretty_print()
        messages = [message]
        result = self.graph.invoke(input={"messages": messages}, config={"recursion_limit": 50})
        return result["messages"][-1]
