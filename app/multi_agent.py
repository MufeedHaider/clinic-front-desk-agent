"""Multi-agent version of the clinic front desk: a SUPERVISOR routes each message to one SPECIALIST.

    START -> guardrail --(emergency)--> END
                 |
                 v
             supervisor  (one cheap LLM call, no tools: answers "info" or "booking")
              |        |
        info  |        | booking
              v        v
        info_agent   booking_agent        <- each has its OWN prompt and ONLY the tools it needs
          ^    |        ^     |
          |    v        |     v
        info_tools   booking_tools         (book_appointment still pauses for the patient's confirmation)

Why split? Each specialist sees a shorter prompt and fewer tools, so it is cheaper and makes fewer
wrong tool calls than one agent holding everything. The cost is one extra routing call per message.
Memory and the confirmation pause work exactly as in the single-agent graph (same checkpointer).
"""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage, trim_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.agent import MAX_HISTORY_MESSAGES, message_text
from app.guardrails import EMERGENCY_REPLY, is_emergency
from app.prompts import ROUTER_PROMPT, booking_prompt, info_prompt

INFO_TOOLS = ("search_clinic_info",)
BOOKING_TOOLS = ("find_patient", "register_patient", "list_available_slots", "book_appointment")
ROUTER_CONTEXT_MESSAGES = 6  # the supervisor only needs the last few turns to decide


class MultiState(MessagesState):
    blocked: bool
    route: str  # "info" or "booking": which specialist handled the latest message


def parse_route(text: str) -> str:
    """Turn the supervisor's reply into a route. Unclear answers fall back to the read-only 'info' agent."""
    return "booking" if "booking" in text.lower() else "info"


def build_multi_graph(llm: BaseChatModel, tools: list, checkpointer=None):
    by_name = {t.name: t for t in tools}
    info_tools = [by_name[n] for n in INFO_TOOLS]
    booking_tools = [by_name[n] for n in BOOKING_TOOLS]
    info_llm = llm.bind_tools(info_tools)
    booking_llm = llm.bind_tools(booking_tools)

    def guardrail(state: MultiState) -> dict:
        last = state["messages"][-1]
        if isinstance(last, HumanMessage) and is_emergency(message_text(last)):
            return {"messages": [AIMessage(content=EMERGENCY_REPLY)], "blocked": True}
        return {"blocked": False}

    def route_after_guardrail(state: MultiState) -> str:
        return END if state["blocked"] else "supervisor"

    def supervisor(state: MultiState) -> dict:
        # Show the router a plain-text transcript of the last few turns (no tool messages needed).
        lines = []
        for m in state["messages"][-ROUTER_CONTEXT_MESSAGES:]:
            if isinstance(m, HumanMessage):
                lines.append(f"Patient: {message_text(m)}")
            elif isinstance(m, AIMessage) and message_text(m).strip():
                lines.append(f"Assistant: {message_text(m)}")
            elif isinstance(m, ToolMessage):
                continue
        reply = llm.invoke([SystemMessage(content=ROUTER_PROMPT), HumanMessage(content="\n".join(lines))])
        return {"route": parse_route(message_text(reply))}

    def pick_specialist(state: MultiState) -> str:
        return "booking_agent" if state["route"] == "booking" else "info_agent"

    def make_agent_node(bound_llm, prompt_fn):
        def node(state: MultiState) -> dict:
            history = trim_messages(
                state["messages"], max_tokens=MAX_HISTORY_MESSAGES, token_counter=len,
                strategy="last", start_on="human",
            )
            return {"messages": [bound_llm.invoke([SystemMessage(content=prompt_fn())] + history)]}
        return node

    graph = StateGraph(MultiState)
    graph.add_node("guardrail", guardrail)
    graph.add_node("supervisor", supervisor)
    graph.add_node("info_agent", make_agent_node(info_llm, info_prompt))
    graph.add_node("info_tools", ToolNode(info_tools))
    graph.add_node("booking_agent", make_agent_node(booking_llm, booking_prompt))
    graph.add_node("booking_tools", ToolNode(booking_tools))

    graph.add_edge(START, "guardrail")
    graph.add_conditional_edges("guardrail", route_after_guardrail, ["supervisor", END])
    graph.add_conditional_edges("supervisor", pick_specialist, ["info_agent", "booking_agent"])
    graph.add_conditional_edges("info_agent", tools_condition, {"tools": "info_tools", END: END})
    graph.add_edge("info_tools", "info_agent")
    graph.add_conditional_edges("booking_agent", tools_condition, {"tools": "booking_tools", END: END})
    graph.add_edge("booking_tools", "booking_agent")

    return graph.compile(checkpointer=checkpointer or InMemorySaver())
