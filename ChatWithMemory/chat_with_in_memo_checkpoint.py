from typing import TypedDict, Annotated
from langgraph.graph import add_messages, StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
import os

load_dotenv(override=True)

# load llm
llm = ChatGroq(api_key=os.getenv('GROQ_API_KEY'),
               model="llama-3.1-8b-instant")

class BasicChatState(TypedDict):
    messages : Annotated[list, add_messages]

# func to get response, used as node
def chatbot(state:BasicChatState):
    return {
        "messages": [llm.invoke(state["messages"])]
    }

#creating in memory instance
memory = MemorySaver()

graph = StateGraph(BasicChatState)

graph.add_node("chatbot", chatbot) 

graph.add_edge("chatbot", END)

graph.set_entry_point("chatbot")

app = graph.compile(checkpointer=memory) ## adding memory

#creating a config to persist/identify user convo
config = {
    "configurable": {
        "thread_id":1
    }
}

while True:
    user_input = input("User: ")
    if user_input in ['exit', 'end']:
        break
    else:
        result = app.invoke({
            "messages":[HumanMessage(content=user_input)]
        }, config=config)
        print("AI:"+ result["messages"][-1].content)