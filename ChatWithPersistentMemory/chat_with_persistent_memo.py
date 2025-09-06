from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, add_messages, END
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
import os

#load env variables
load_dotenv(override=True)

#create llm instance
llm = ChatGroq(
    api_key= os.getenv('GROQ_API_KEY'),
    model="llama-3.1-8b-instant")


## create sqlite3 connection
sqlite_conn = sqlite3.connect("persistent_checkpoint.sqlite", check_same_thread=False)

#create persistent sqlite memory instance 
memory = SqliteSaver(sqlite_conn)

class BasicChatState(TypedDict):
    messages : Annotated[list, add_messages]

def chatbot(state:BasicChatState):
    return {
        "messages": [llm.invoke(state["messages"])]
    }

#creating graph
graph = StateGraph(BasicChatState)

graph.add_node("chatbot", chatbot)

graph.add_edge("chatbot", END)

graph.set_entry_point("chatbot")

app = graph.compile(checkpointer=memory)

#creating configutation to manage threads
config = {
    "configurable": {
        "thread_id":1
    }
}

while True:
    user_input = input("User: ") ## user input
    if user_input in ['exit', 'break']:
        break
    else:
        response = app.invoke(
            {
                "messages" : [HumanMessage(content = user_input)]
            },
            config=config
        )
        print("AI:"+response["messages"][-1].content)
