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
   - Rename `.env.example` to `.env`.
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

## Example Screenshots

### Web Search and PDF RAG
![Web Search and PDF RAG](./screenshots/web_search_rag.png)

### URL Summarization
![URL Summarization](./screenshots/url_summarization.png)

## Future Enhancements

- Support for additional document formats (e.g., Word files).
- Advanced summarization controls, including length and format preferences.
- Real-time language translation for multilingual support.

## Contributions

Contributions are welcome! Please feel free to open issues, submit feature requests, or create pull requests.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

Replace `"https://github.com/your-username/langchain-multitool-app.git"` with your actual repository URL, and adjust paths or file names if needed. This README provides a clean overview of your project’s capabilities and setup instructions. Let me know if you'd like to include additional information!
