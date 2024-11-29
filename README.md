# LangChain Multitool App

A versatile Streamlit application that integrates LangChain-powered tools for intelligent web search, retrieval-augmented generation (RAG) with PDF support, and content summarization from YouTube videos or web pages. This app leverages the Groq API to deliver seamless, context-aware answers, making it a powerful assistant for research, knowledge extraction, and quick content summarization.

## Features

- **Intelligent Web Search**: Uses DuckDuckGo for web searches, providing real-time responses.
- **RAG from PDFs**: Retrieves and answers questions using content from uploaded PDF files through FAISS-based vector search.
- **Content Summarization**: Summarizes text from YouTube videos and web pages, delivering concise or detailed summaries as chosen by the user.
- **Configurable Responses**: Users can select response detail levels, save chat histories, and adjust session preferences.
  
## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/langchain-multitool-app.git
   cd langchain-multitool-app
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up API keys**:
   - Add your **Groq API Key** and **HuggingFace Token** (for embeddings) to the `.env` file.

## Usage

1. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

2. **Features available in the app**:
   - **Web Search**: Ask questions or explore topics using the web search powered by LangChain's DuckDuckGo integration.
   - **RAG with PDFs**: Upload one or more PDF files to retrieve answers based on their content.
   - **URL Summarization**: Enter any YouTube or webpage URL to generate summaries.

## Project Structure

- **app.py**: Main Streamlit app file containing all interface code.
- **requirements.txt**: Lists required Python packages.
- **.env.example**: Template for environment variables file, which includes placeholders for API keys.
- **helpers/**: Directory for helper functions (e.g., for loading PDF files or managing session history).


## Contributions

Contributions are welcome! Please feel free to open issues, submit feature requests, or create pull requests.

## App Link

Explore the live application here: [Multifunctional AI-Powered Information Retrieval and Summarization App](https://anishaman6206-langchain-multitool-app-app-d8mfu1.streamlit.app/)

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

