import streamlit as st
from langchain_ollama import OllamaLLM

# --- Page Configuration ---
st.set_page_config(page_title="Gemma 3 Chat", page_icon="🤖")

st.title("🚀 Gemma 3 Chat App")
st.markdown("Running locally via Ollama")

# --- Initialize Model ---
# We use 'gemma3:latest' to match your 'ollama list' exactly
try:
    llm = OllamaLLM(
        model="gemma3:latest",
        base_url="http://localhost:11434",
        temperature=0.7
    )
except Exception as e:
    st.error(f"Failed to initialize model: {e}")

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask Gemma 3 something..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Gemma is thinking..."):
            try:
                # Direct call to the model
                response = llm.invoke(prompt)
                st.markdown(response)
                # Add assistant response to history
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Ollama Error: {e}")
                st.info("Check if Ollama is running and you have pulled gemma3.")