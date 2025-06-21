import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Constants
SELECTED_MODEL = "deepseek-r1:1.5b"
llm_engine = ChatOllama(model=SELECTED_MODEL, base_url="http://localhost:11434", temperature=0.3)

# CSS Styling
st.markdown("""
    <style>
        .stApp { background-color: #0E1117; color: #FFFFFF; }
        .stChatInput input { background-color: #1E1E1E !important; color: #FFFFFF !important; border: 1px solid #3A3A3A !important; }
        .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) { background-color: #1E1E1E !important; color: #E0E0E0 !important; }
        .stChatMessage[data-testid="stChatMessage"]:nth-child(even) { background-color: #2A2A2A !important; color: #F0F0F0 !important; }
        h1, h2, h3 { color: #1DA1F2 !important; }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("DiagnoSphere Chatbot")
st.caption("Your AI partner for medical diagnosis, home remedies, and report analysis")

# Sidebar
with st.sidebar:
    st.markdown("### Model Capabilities")
    st.markdown("- 🐞 Diagnosing Assistant\n- 📝 Report Analyzer\n- 💡 Home Remedies")
    st.markdown("Built with [Ollama](https://ollama.ai/) | [LangChain](https://python.langchain.com/)")

# System Prompt
SYSTEM_PROMPT = SystemMessagePromptTemplate.from_template(
    "You are an expert AI medical assistant. Provide diagnoses based on symptoms, suggest home remedies, "
    "and analyze medical reports. Recommend a doctor if needed. Respond in English."
)

# Session state for chat history & think visibility
if "message_log" not in st.session_state:
    st.session_state.message_log = [{"role": "ai", "content": "Hello! Describe your symptoms, or upload a report for analysis."}]
if "think_visibility" not in st.session_state:
    st.session_state.think_visibility = {}

# Display chat history
for i, message in enumerate(st.session_state.message_log):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("think"):
            if f"think_{i}" not in st.session_state.think_visibility:
                st.session_state.think_visibility[f"think_{i}"] = False  # Default to hidden
            
            if st.button("Toggle AI Thinking", key=f"think_button_{i}"):
                st.session_state.think_visibility[f"think_{i}"] = not st.session_state.think_visibility[f"think_{i}"]
                st.rerun()  # Rerun to update visibility
            
            if st.session_state.think_visibility[f"think_{i}"]:
                st.markdown(message["think"])

# Function to generate AI response
def generate_ai_response(prompt_chain):
    processing_pipeline = prompt_chain | llm_engine | StrOutputParser()
    return processing_pipeline.invoke({})

# Build chat prompt sequence
def build_prompt_chain():
    prompt_sequence = [SYSTEM_PROMPT]
    for msg in st.session_state.message_log:
        if msg["role"] == "user":
            prompt_sequence.append(HumanMessagePromptTemplate.from_template(msg["content"]))
        elif msg["role"] == "ai":
            prompt_sequence.append(AIMessagePromptTemplate.from_template(msg["content"]))
    return ChatPromptTemplate.from_messages(prompt_sequence)

# Handle user queries
user_query = st.chat_input("Enter your symptoms or question...")

if user_query:
    st.session_state.message_log.append({"role": "user", "content": user_query})

    with st.spinner("Analyzing..."):
        prompt_chain = build_prompt_chain()
        ai_response = generate_ai_response(prompt_chain)

    # Separate the "think" part from the main response
    think_start = ai_response.find("<think>")
    think_end = ai_response.find("</think>") + 8  # Include </think> tag
    if think_start != -1 and think_end != -1:
        think_part = ai_response[think_start:think_end]
        main_response = ai_response[:think_start] + ai_response[think_end:]
    else:
        think_part = None
        main_response = ai_response

    # Store chatbot response
    st.session_state.message_log.append({"role": "ai", "content": main_response, "think": think_part})

    st.rerun()
