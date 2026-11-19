"""The agent workflow, built as a LangGraph state graph.

    START -> guardrail --(emergency)--> END
                 |
                 v
               agent  <------+
                 |           |
          (wants a tool?)    |
           yes |   no -> END |
               v             |
             tools ----------+      (book_appointment pauses here for confirmation)

- guardrail: blocks emergencies before the LLM ever sees them
- agent:     the LLM reasons and decides whether to call a tool (the ReAct loop)
- tools:     runs the chosen tool and feeds the result back to the agent
- memory:    a checkpointer saves the conversation per thread_id, so follow-up messages have context
"""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, trim_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command

from app.guardrails import EMERGENCY_REPLY, is_emergency
from app.prompts import system_prompt

MAX_HISTORY_MESSAGES = 40  # short-term memory window sent to the LLM


class ClinicState(MessagesState):
    blocked: bool


def message_text(msg: BaseMessage) -> str:
    """Models return either a string or a list of content parts; normalise to plain text."""
    if isinstance(msg.content, str):
        return msg.content
    return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in msg.content)


def build_graph(llm: BaseChatModel, tools: list, checkpointer=None):
    llm_with_tools = llm.bind_tools(tools)

    def guardrail(state: ClinicState) -> dict:
        last = state["messages"][-1]
        if isinstance(last, HumanMessage) and is_emergency(message_text(last)):
            return {"messages": [AIMessage(content=EMERGENCY_REPLY)], "blocked": True}
        return {"blocked": False}

    def route_after_guardrail(state: ClinicState) -> str:
        return END if state["blocked"] else "agent"

    def agent(state: ClinicState) -> dict:
        history = trim_messages(
            state["messages"], max_tokens=MAX_HISTORY_MESSAGES, token_counter=len,
            strategy="last", start_on="human",
        )
        response = llm_with_tools.invoke([SystemMessage(content=system_prompt())] + history)
        return {"messages": [response]}

    graph = StateGraph(ClinicState)
    graph.add_node("guardrail", guardrail)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(tools))

    graph.add_edge(START, "guardrail")
    graph.add_conditional_edges("guardrail", route_after_guardrail, ["agent", END])
    graph.add_conditional_edges("agent", tools_condition, ["tools", END])
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=checkpointer or InMemorySaver())


class ClinicAgent:
    """A small service wrapper the API calls: send a message, or answer a pending confirmation."""

    def __init__(self, graph):
        self.graph = graph

    def _config(self, thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}}

    def pending(self, thread_id: str) -> dict | None:
        snapshot = self.graph.get_state(self._config(thread_id))
        interrupts = getattr(snapshot, "interrupts", ()) or ()
        return interrupts[0].value if interrupts else None

    def _result(self, thread_id: str) -> dict:
        pending = self.pending(thread_id)
        messages = self.graph.get_state(self._config(thread_id)).values.get("messages", [])
        reply = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and message_text(msg).strip():
                reply = message_text(msg)
                break
        if pending:
            reply = pending.get("question", "Please confirm.")
        handled_by = self.graph.get_state(self._config(thread_id)).values.get("route")  # multi-agent mode only
        return {"thread_id": thread_id, "reply": reply, "pending_confirmation": pending, "handled_by": handled_by}

    def chat(self, thread_id: str, message: str) -> dict:
        if self.pending(thread_id):
            raise RuntimeError("A booking is waiting for confirmation. Answer it first.")
        self.graph.invoke({"messages": [HumanMessage(content=message)]}, self._config(thread_id))
        return self._result(thread_id)

    def confirm(self, thread_id: str, approved: bool) -> dict:
        if not self.pending(thread_id):
            raise LookupError("Nothing is waiting for confirmation on this thread.")
        self.graph.invoke(Command(resume={"approved": approved}), self._config(thread_id))
        return self._result(thread_id)
