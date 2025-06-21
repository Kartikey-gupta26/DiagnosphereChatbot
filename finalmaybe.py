import streamlit as st
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_core.output_parsers import StrOutputParser

# Constants
PDF_STORAGE_PATH = 'document_store/'
SELECTED_MODEL = "deepseek-r1:1.5b"
EMBEDDING_MODEL = OllamaEmbeddings(model=SELECTED_MODEL)
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
LANGUAGE_MODEL = OllamaLLM(model=SELECTED_MODEL)

# CSS Styling
st.markdown("""
    <style>
        .stApp { background-color: #0E1117; color: #FFFFFF; }
        .stChatInput input { background-color: #1E1E1E !important; color: #FFFFFF !important; border: 1px solid #3A3A3A !important; }
        .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) { background-color: #1E1E1E !important; color: #E0E0E0 !important; }
        .stChatMessage[data-testid="stChatMessage"]:nth-child(even) { background-color: #2A2A2A !important; color: #F0F0F0 !important; }
        .stFileUploader { background-color: #1E1E1E; border: 1px solid #3A3A3A; padding: 15px; }
        h1, h2, h3 { color: #1DA1F2 !important; }
    </style>
""", unsafe_allow_html=True)

# UI Elements
st.title("DiagnoSphere Chatbot")
st.caption("Your AI partner for medical diagnosis, home remedies, and report analysis")

# Sidebar
with st.sidebar:
    st.markdown("### Model Capabilities")
    st.markdown("- 🐞 Diagnosing Assistant\n- 📝 Report Analyzer\n- 💡 Home Remedies")
    st.markdown("Built with [Ollama](https://ollama.ai/) | [LangChain](https://python.langchain.com/)")
    st.markdown("""
    <style>
        .chatbot-btn {
            position: relative;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            padding: 12px 20px;
            background-color: #1DA1F2;
            color: white !important;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            font-size: 18px;
            text-decoration: none !important;
            box-shadow: 0 2px 10px rgba(29, 161, 242, 0.3);
            transition: all 0.2s ease;
            cursor: pointer;
            width: 100%;
            margin: 10px 0;
        }
        .chatbot-btn:hover {
            background-color: #0d8bf0;
            color: white !important;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(29, 161, 242, 0.4);
        }
        .chatbot-btn:active {
            transform: translateY(0);
        }
        /* Target the link specifically to override Streamlit styles */
        a.chatbot-btn, a.chatbot-btn:hover, a.chatbot-btn:visited {
            color: white !important;
            text-decoration: none !important;
        }
    </style>
    <a href="http://127.0.0.1:5000/" target="_self" class="chatbot-btn">
        📷 Image Classifier
    </a>
    """, unsafe_allow_html=True)
    # Toggle for report analysis
    analyze_report = st.toggle("Analyze Uploaded Report", value=False)

SYSTEM_PROMPT = SystemMessagePromptTemplate.from_template(
    "You are an expert AI medical assistant trained in diagnosing diseases. "
    "Provide accurate and well-explained diagnoses based on symptoms. "
    "When applicable, suggest helpful home remedies. "
    "If a condition is serious, recommend consulting a healthcare professional. Respond only in English."
)

# Initialize chat engine
llm_engine = ChatOllama(model=SELECTED_MODEL, base_url="http://localhost:11434", temperature=0.3)

# Session state for chat history & "thinking" visibility
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

# PDF Processing Functions
def save_uploaded_file(uploaded_file):
    file_path = PDF_STORAGE_PATH + uploaded_file.name
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path

def load_pdf_documents(file_path):
    return PDFPlumberLoader(file_path).load()

def chunk_documents(raw_documents):
    return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(raw_documents)

def index_documents(document_chunks):
    DOCUMENT_VECTOR_DB.add_documents(document_chunks)

def find_related_documents(query):
    return DOCUMENT_VECTOR_DB.similarity_search(query)

def generate_answer(user_query, context_documents):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    conversation_prompt = ChatPromptTemplate.from_template(
        "Query: {user_query}\nContext: {document_context}\nAnswer:"
    )
    response_chain = conversation_prompt | LANGUAGE_MODEL
    return response_chain.invoke({"user_query": user_query, "document_context": context_text})

# Only show file uploader when the toggle is ON
if analyze_report:
    uploaded_pdf = st.file_uploader("Upload your medical report (PDF)", type="pdf")

    # Process uploaded PDF only if toggle is enabled and a file is uploaded
    if uploaded_pdf:
        saved_path = save_uploaded_file(uploaded_pdf)
        raw_docs = load_pdf_documents(saved_path)
        processed_chunks = chunk_documents(raw_docs)
        index_documents(processed_chunks)
        
        st.success("✅ Document uploaded and processed! Ask questions about it below.")
        
        doc_query = st.chat_input("Ask about the uploaded report...")

        if doc_query:
            with st.chat_message("user"):
                st.write(doc_query)
            
            with st.spinner("Analyzing report..."):
                relevant_docs = find_related_documents(doc_query)
                ai_response = generate_answer(doc_query, relevant_docs)
                
            with st.chat_message("assistant", avatar="🤖"):
                st.write(ai_response)
