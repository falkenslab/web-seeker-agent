import operator
from typing import TypedDict, Annotated

from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.errors import GraphRecursionError

import agent

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add] # Crea una lista de mensajes que se puede concatenar con el operador +


class Agent:

    _current_state: AgentState = None

    def __init__(self, model, tools, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)         # Nodo de llamada al modelo
        graph.add_node("action", self.take_action)      # Nodo de ejecución de herramientas (acciones)
        graph.add_conditional_edges(
            "llm",                                      # La arista condicional sale del nodo del modelo
            self.exists_action,                         # Función que decide si se debe ir al nodo de acción o terminar
            {True: "action", False: END}
        )                                               # Si el modelo decide llamar a una herramienta, ir al nodo de acción; si no, terminar
        graph.add_edge("action", "llm")                 # Después de ejecutar una acción, volver al modelo      
        graph.set_entry_point("llm")                    # El punto de entrada del grafo es la llamada al modelo
        self.graph = graph.compile()                    # Compilar el grafo para su ejecución
        self.tools = {t.name: t for t in tools}         # Mapeo de herramientas por nombre
        self.model = model.bind_tools(tools)            # Asociar las herramientas al modelo

    def exists_action(self, state: AgentState) -> bool:
        self._current_state = state
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def call_openai(self, state: AgentState) -> AgentState:
        self._current_state = state
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
            self.system = ""  # Only use system prompt once
        message = self.model.invoke(messages)
        return {'messages': [message]}

    def take_action(self, state: AgentState) -> AgentState:
        self._current_state = state
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"\nEjecutando acción: {t}\n")
            if not t['name'] in self.tools:      # check for bad tool name from LLM
                print("\n ....nombre de tool no válida....")
                result = "nombre de tool no válida, reintentar"  # instruct LLM to retry if bad
            else:
                result = self.tools[t['name']].invoke(t['args'])
            message = ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result))
            message.pretty_print()
            results.append(message)
        return {'messages': results}
    
    def ask(self, question: str) -> AnyMessage:
        message = HumanMessage(content=question)
        message.pretty_print()
        messages = [message]
        result = self.graph.invoke(input={"messages": messages}, config={"recursion_limit": 50})
        return result['messages'][-1]
