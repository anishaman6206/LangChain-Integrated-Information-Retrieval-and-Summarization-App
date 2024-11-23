import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain.document_loaders import PyPDFLoader
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain.chains.summarize import load_summarize_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import validators
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# Streamlit setup
st.title("Unified LangChain Application: Web Search, Conversational RAG, and URL Summarization")

# Display an information section about the Groq API key
st.markdown("### Groq API Key Information")
st.info(
    "To use this app features, you need a **Groq API key**.\n"
    "If you don't have a Groq API key, you can follow these steps to get one:\n"
    "1. Go to the [Groq website](https://www.groq.com).\n"
    "2. Sign up or log in to your Groq account.\n"
    "3. Navigate to the **API section** in your account settings.\n"
    "4. Generate a new API key and copy it.\n\n"
    "Once you have your API key, enter it to start using the app."
)

# Sidebar settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")
app_mode = st.sidebar.radio("Choose Mode", ["Web Search", "Conversational RAG with PDF Uploads", "Summarize URL"])
response_detail = st.sidebar.radio("Response Detail Level:", ["Concise", "Detailed"], index=0)
save_history = st.sidebar.checkbox("Enable Saving Responses")

# Initialize chat history and saved responses
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hi! How can I assist you today?"}]
if "saved_responses" not in st.session_state:
    st.session_state["saved_responses"] = []

# Shared Embeddings for PDF RAG
HF_TOKEN="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"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# Define common functions
def display_chat_history():
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

def save_response(response):
    if save_history:
        if st.button("Save this response"):
            st.session_state.saved_responses.append(response)
            st.sidebar.success("Response saved!")

# Initialize LLM based on API key
if api_key:
    # Define LLM model and detail settings
    model_name = "Llama3-8b-8192" if app_mode in ["Web Search", "Summarize URL"] else "Gemma2-9b-It"
    max_tokens = 300 if response_detail == "Concise" else 1000
    llm = ChatGroq(groq_api_key=api_key, model_name=model_name, streaming=True, max_tokens=max_tokens)

    # Mode 1: Web Search
    if app_mode == "Web Search":
        # Initialize search tools
        arxiv_wrapper = ArxivAPIWrapper(top_k_results=3, doc_content_chars_max=500)
        arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)
        wiki_wrapper = WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=500)
        wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)
        search = DuckDuckGoSearchRun(name="Search")
        
        # Initialize search agent
        tools = [search, arxiv, wiki]
        search_agent = initialize_agent(
            tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handling_parsing_errors=True, max_iterations=5
        )

        display_chat_history()
        if prompt := st.chat_input("Ask me something..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)

            # Display assistant response
            with st.chat_message("assistant"):
                st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
                #prompt_with_context = "\n".join([msg["content"] for msg in st.session_state.messages])
                response = search_agent.run(prompt, callbacks=[st_cb])
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.write(response)
                save_response(response)

    # Mode 2: Conversational RAG with PDF Uploads
    elif app_mode == "Conversational RAG with PDF Uploads":
        # Chat interface
        session_id = st.text_input("Session ID", value="default_session")
    
        # Manage chat history in Streamlit session state
        if 'store' not in st.session_state:
            st.session_state.store = {}
    
        # Upload PDF files
        uploaded_files = st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=True)
    
        # Process uploaded PDFs
        if uploaded_files:
            documents = []
            for uploaded_file in uploaded_files:
                with open(f"./{uploaded_file.name}", "wb") as file:
                    file.write(uploaded_file.getvalue())
    
                loader = PyPDFLoader(f"./{uploaded_file.name}")
                docs = loader.load()
                documents.extend(docs)
    
            # Split and create embeddings for the documents
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
            splits = text_splitter.split_documents(documents)
    
            # Initialize FAISS vector store
            vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
            retriever = vectorstore.as_retriever()
    
            # Set up question contextualization
            contextualize_q_system_prompt = (
                "Given a chat history and the latest user question "
                "which might reference context in the chat history, "
                "formulate a standalone question which can be understood "
                "without the chat history. Do NOT answer the question, "
                "just reformulate it if needed and otherwise return it as is."
            )
            contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
    
            history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    
            # Configure answer prompt
            if response_detail=='Concise':

             system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Use three sentences maximum and keep the answer concise."
                "\n\n"
                "{context}"
            )
            else:
             system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Keep the answer detailed."
                "\n\n"
                "{context}"
             )     
            qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
    
            question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
            # Function to manage session history
            def get_session_history(session: str) -> BaseChatMessageHistory:
                if session_id not in st.session_state.store:
                    st.session_state.store[session_id] = ChatMessageHistory()
                return st.session_state.store[session_id]
    
            conversational_rag_chain = RunnableWithMessageHistory(
                rag_chain,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )
    
            # Get user input and respond
            user_input = st.text_input("Your question:")
            if user_input:
                session_history = get_session_history(session_id)
                response = conversational_rag_chain.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": session_id}}
                )
                st.write("Assistant:", response['answer'])
                st.write("Last Question:", user_input)
                st.write("Last Response:", response['answer'])
                #st.write("Chat History:", session_history.messages)
                save_response(response['answer'])

    # Mode 3: Summarize URL
    elif app_mode == "Summarize URL":
        generic_url = st.text_input("Enter URL for Summarization (YouTube or Website)")
        
        if st.button("Summarize Content"):
            if not validators.url(generic_url):
                st.error("Please enter a valid URL (YouTube or website).")
            else:
                try:
                    with st.spinner("Fetching and summarizing content..."):
                        # Load content from YouTube or website
                        if "youtube.com" in generic_url:
                            loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=False)
                        else:
                            loader = UnstructuredURLLoader(
                                urls=[generic_url],
                                ssl_verify=False,
                                headers={"User-Agent": "Mozilla/5.0"}
                            )
                        docs = loader.load()

                        # Summarization prompt and chain
                        if response_detail=='Concise':
                         prompt_template = "Provide a concise summary of the following content:\nContent:{text}\n"
                        else:
                          prompt_template = "Provide a detailed summary of the following content:\nContent:{text}\n"
                        prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
                        chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                        output_summary = chain.run(docs)

                        st.success(output_summary)
                        save_response(output_summary)
                except Exception as e:
                    st.exception(f"Exception: {e}")

else:
    st.warning("Please enter the Groq API Key to proceed.")

# Display saved responses in the sidebar
if save_history and st.session_state.saved_responses:
    st.sidebar.subheader("Saved Responses")
    for i, saved_response in enumerate(st.session_state.saved_responses, 1):
        with st.sidebar.expander(f"Response {i}"):
            st.write(saved_response)
