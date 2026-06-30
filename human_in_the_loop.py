from dotenv import load_dotenv

_ = load_dotenv()

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_community.chat_models import ChatZhipuAI
from langchain_tavily import TavilySearch

from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

from uuid import uuid4
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage

"""
In previous examples we've annotated the `messages` state key
with the default `operator.add` or `+` reducer, which always
appends new messages to the end of the existing messages array.

Now, to support replacing existing messages, we annotate the
`messages` key with a customer reducer function, which replaces
messages with the same `id`, and appends them otherwise.
"""
def reduce_messages(left: list[AnyMessage], right: list[AnyMessage]) -> list[AnyMessage]:
    # assign ids to messages that don't have them
    for message in right:
        if not message.id:
            message.id = str(uuid4())
    # merge the new messages with the existing messages
    merged = left.copy()
    for message in right:
        for i, existing in enumerate(merged):
            # replace any existing messages with the same id
            if existing.id == message.id:
                merged[i] = message
                break
        else:
            # append any new messages to the end
            merged.append(message)
    return merged

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], reduce_messages]

tool = TavilySearch(max_results=2)

class Agent:
    def __init__(self, model, tools, system="", checkpointer=None):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges("llm", self.exists_action, {True: "action", False: END})
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile(
            checkpointer=checkpointer,
            interrupt_before=["action"]
        )
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def call_openai(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}

    def exists_action(self, state: AgentState):
        print(state)
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("Back to the model!")
        return {'messages': results}

prompt = """You are a smart research assistant. Use the search engine to look up information. \
You are allowed to make multiple calls (either together or in sequence). \
Only look up information when you are sure of what you want. \
If you need to look up some information before asking a follow up question, you are allowed to do that!
"""
model = ChatZhipuAI(
    temperature=0,
    model="glm-4-flash"
)
abot = Agent(model, [tool], system=prompt, checkpointer=memory)

messages = [HumanMessage(content="Whats the weather in SF?")]
thread = {"configurable": {"thread_id": "1"}}
for event in abot.graph.stream({"messages": messages}, thread):
    for v in event.values():
        print(v)

abot.graph.get_state(thread)

abot.graph.get_state(thread).next

for event in abot.graph.stream(None, thread):
    for v in event.values():
        print(v)

abot.graph.get_state(thread)

abot.graph.get_state(thread).next

messages = [HumanMessage("Whats the weather in LA?")]
thread = {"configurable": {"thread_id": "2"}}
for event in abot.graph.stream({"messages": messages}, thread):
    for v in event.values():
        print(v)
while abot.graph.get_state(thread).next:
    print("\n", abot.graph.get_state(thread),"\n")
    _input = input("proceed?")
    if _input != "y":
        print("aborting")
        break
    for event in abot.graph.stream(None, thread):
        for v in event.values():
            print(v)

# Modify State
messages = [HumanMessage("Whats the weather in LA?")]
thread = {"configurable": {"thread_id": "3"}}
for event in abot.graph.stream({"messages": messages}, thread):
    for v in event.values():
        print(v)

abot.graph.get_state(thread).next

abot.graph.get_state(thread)

current_values = abot.graph.get_state(thread)

current_values.values['messages'][-1]

current_values.values['messages'][-1].tool_calls

_id = current_values.values['messages'][-1].tool_calls[0]['id']
current_values.values['messages'][-1].tool_calls = [
    {'name': 'tavily_search',
  'args': {'query': 'current weather in Louisiana'},
  'id': _id}
]

abot.graph.update_state(thread, current_values.values)

abot.graph.get_state(thread)

for event in abot.graph.stream(None, thread):
    for v in event.values():
        print(v)

# Time Travel
states = []
for state in abot.graph.get_state_history(thread):
    print(state)
    print('--')
    states.append(state)

to_replay = states[-3]
to_replay

for event in abot.graph.stream(None, to_replay.config):
    for k, v in event.items():
        print(v)

# Go back in time and edit
to_replay

_id = to_replay.values['messages'][-1].tool_calls[0]['id']
to_replay.values['messages'][-1].tool_calls = [{'name': 'tavily_search',
  'args': {'query': 'current weather in LA, accuweather'},
  'id': _id}]

branch_state = abot.graph.update_state(to_replay.config, to_replay.values)

for event in abot.graph.stream(None, branch_state):
    for k, v in event.items():
        if k != "__end__":
            print(v)

# Add message to a state at a given time
to_replay

_id = to_replay.values['messages'][-1].tool_calls[0]['id']

state_update = {"messages": [ToolMessage(
    tool_call_id=_id,
    name="tavily_search",
    content="54 degree celcius",
)]}

branch_and_add = abot.graph.update_state(
    to_replay.config, 
    state_update, 
    as_node="action")

for event in abot.graph.stream(None, branch_and_add):
    for k, v in event.items():
        print(v)

# Extra Practice

## Build a small graph
'''
Define a simple 2 node graph with the following state:
-`lnode`: last node
-`scratch`: a scratchpad location
-`count` : a counter that is incremented each step
'''
class AgentState(TypedDict):
    lnode: str
    scratch: str
    count: Annotated[int, operator.add]

def node1(state: AgentState):
    print(f"node1, count:{state['count']}")
    return {"lnode": "node_1",
            "count": 1,
           }
def node2(state: AgentState):
    print(f"node2, count:{state['count']}")
    return {"lnode": "node_2",
            "count": 1,
           }

'''The graph goes N1->N2->N1... but breaks after count reaches 3.'''
def should_continue(state):
    return state["count"] < 3

builder = StateGraph(AgentState)
builder.add_node("Node1", node1)
builder.add_node("Node2", node2)

builder.add_edge("Node1", "Node2")
builder.add_conditional_edges("Node2", 
                              should_continue, 
                              {True: "Node1", False: END})
builder.set_entry_point("Node1")

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

thread = {"configurable": {"thread_id": str(1)}}
graph.invoke({"count":0, "scratch":"hi"},thread)

from pprint import pprint

state = graph.get_state(thread)
pprint(state._asdict(), width=1, sort_dicts=False)

for state in graph.get_state_history(thread):
    pprint(state._asdict(), width=1, sort_dicts=False)
    print("--")

states = []
for state in graph.get_state_history(thread):
    states.append(state.config)
    print(state.config, state.values['count'])

states[-3]

state = graph.get_state(states[-3])
pprint(state._asdict(), width=1, sort_dicts=False)

## Go Back in Time
graph.invoke(None, states[-3])

thread = {"configurable": {"thread_id": str(1)}}
for state in graph.get_state_history(thread):
    print(state.config, state.values['count'])

thread = {"configurable": {"thread_id": str(1)}}
for state in graph.get_state_history(thread):
    print(state,"\n")

## Modify State
thread2 = {"configurable": {"thread_id": str(2)}}
graph.invoke({"count":0, "scratch":"hi"},thread2)

from IPython.display import Image

Image(graph.get_graph().draw_png())

states2 = []
for state in graph.get_state_history(thread2):
    states2.append(state.config)
    # print(state.config, state.values['count'])
    print(state)

save_state = graph.get_state(states2[-3])
pprint(save_state._asdict(), width=1, sort_dicts=False)

save_state.values["count"] = -3
save_state.values["scratch"] = "hello"
pprint(save_state._asdict(), width=1, sort_dicts=False)

graph.update_state(thread2,save_state.values)

for i, state in enumerate(graph.get_state_history(thread2)):
    # pprint(state._asdict(), width=1, sort_dicts=False)
    # print("--")
    print(state)

## Try again with as_node
"""
When writing using update_state(), you want to define to the graph logic which node should be assumed as the writer. What this does is allow th graph logic to find the node on the graph. After writing the values, the next() value is computed by travesing the graph using the new state. In this case, the state we have was written by Node1. The graph can then compute the next state as being Node2. Note that in some graphs, this may involve going through conditional edges! Let's try this out.
"""
graph.update_state(thread2,save_state.values, as_node="Node1")

for i, state in enumerate(graph.get_state_history(thread2)):
    print(state)

"""invoke will run from the current state if not given a particular thread_ts. This is now the entry that was just added."""
graph.invoke(None,thread2)

'''Print out the state history, notice the scratch value change on the latest entries.'''
for state in graph.get_state_history(thread2):
    print(state)
