import uuid
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import TypedDict, Annotated, Literal

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain.chat_models import init_chat_model
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

load_dotenv()

llm = init_chat_model("openai:gpt-4.1-mini")

K_Base = []

vector_store = InMemoryVectorStore(OpenAIEmbeddings(model = "text-embedding-3-small"))
vector_store.add_documents([Document(page_content = text) for text in K_Base])

class IntentClassifier(BaseModel):
	message_intent: Literal["chat", "knowledge", "code"] = Field(..., description = ["Classify whether the user wants to chat, ask for knowledge or change code in the project"])

class State(TypedDict):
	messages: Annotated[list, add_messages]
	message_intent:  str | None

def classify_intent(state: State):
	structured_llm = llm.with_structured_output(IntentClassifier)
	result = structured_llm.invoke([
		{"role": "system", "content": "Determine/Classify whether the user wants to chat ('chat'), retrieve knowledge ('knowledge') or change code ('code')"},
		{"role": "user", "content": state["messages"][-1].content}
	])

	return {"message_intent": result.message_intent}

def prompt_llm_chat(state: State):
	messages = [{"role": "system", "content": "You are a talkative chatbot for fun. Be nice"}] + state["messages"]
	response = llm.invoke(messages)
	return {"messages": [{"role": "assistant", "content": response.content}]}

def prompt_llm_rag(state: State):
	query = state["messages"][-1].content
	documents = vector_store.similarity_search(query, k = 3)
	context = "".join(f" -{doc.page_content}" for doc in documents)

	messages = [{"role": "system", "content": f"You are a RAG agent. Answer the user using only the context below. If the answer is not in the context then say you dont know.\n\nContext:\n {context}"}] + state["messages"]
	response = llm.invoke(messages)
	return {"messages": [{"role": "assistant", "content": response.content}]}

def prompt_llm_code(state: State):
	messages = [{"role": "system", "content": "No matter what, always say 'I am the Code Bot'"}] + state["messages"]
	response = llm.invoke(messages)
	return {"messages": [{"role": "assistant", "content": response.content}]}

graph_builder = StateGraph(State)

graph_builder.add_node("classifier", classify_intent)
graph_builder.add_node("chat_agent", prompt_llm_chat)
graph_builder.add_node("rag_agent", prompt_llm_rag)
graph_builder.add_node("coding_agent", prompt_llm_code)

graph_builder.add_edge(START, "classifier")
graph_builder.add_conditional_edge("classifier", lambda state: state["message_intent"], {"chat": "chat_agent", "knowledge": "rag_agent", "code": "coding_agent"})
graph_builder.add_edge("chat_agent", END)
graph_builder.add_edge("rag_agent", END)
graph_builder.add_edge("code_agent", END)

checkpointer = InMemorySaver()
graph = graph_builder.compile(checkpointer = checkpointer)

graph.get_graph().draw_mermaid_png(output_file_path = "graph.png")

config = {"configurable": {"thread_id": uuid.uuid4()}}

while True:
	user_message = input("Enter message: ")
	result = graph.invoke({"messages": [{"role": "user", "content": user_message}]}, config = config)
	print(result["messages"][-1].content)
