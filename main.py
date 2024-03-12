import streamlit as st
from maya import maya_agent_executor

from langchain_core.messages import HumanMessage, AIMessage


def get_chat_history(min: int, max: int):
    if "messages" not in st.session_state:
        return []
    if min < 0 or max > (len(st.session_state.messages) - 1):
        raise Exception("Input out of range.")
    message_history = [(HumanMessage(content=message["content"]) if message["role"] == "user" else AIMessage(content=message["content"])) for message in st.session_state.messages[min:max]]
    return message_history
    

def get_chatbot_response(input: str) -> str:
    response = maya_agent_executor.invoke({
        "input": input,
        "chat_history": get_chat_history(
            min=0,
            max=len(st.session_state.messages)-2
        ) if len(st.session_state.messages) > 1 else []
    })
    print(response)
    return response["output"]


st.title("Maya agente")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Mensaje para Maya..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    response = get_chatbot_response(prompt)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
        
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
