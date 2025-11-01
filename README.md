# PDF Chatbot

A PDF Question-Answering chatbot that allows users to upload PDFs, generates embeddings using HuggingFace models, stores them in Pinecone, and answers queries using Groq LLM. 

---

## Features

- Extracts text from PDF files.
- Splits text into manageable chunks.
- Generates embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
- Stores embeddings in Pinecone for efficient retrieval.
- Answers user queries strictly based on uploaded PDF content using Groq LLM.
- Cleans up Pinecone data after the chat session ends.

---

## Requirements

- Python 3.11
- Pinecone account & API key
- Groq account & API key
- Libraries:
  ```bash
  pip install -r requirements.txt



## Usage

```bash
# 1. Clone the repository
git clone https://github.com/anubhavshakyaaa/pdf-chatbot.git
cd pdf-chatbot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create a .env file in the root directory with the following variables
# (replace with your actual keys)
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
PINECONE_INDEX_NAME=your_index_name
GROQ_API_KEY=your_groq_api_key

# 4. Place your PDF files in a folder, e.g., `files/`

# 5. Run the chatbot
python main.py

# 6. Interact with the chatbot:
# - Ask questions about the uploaded PDFs.
# - Type 'exit' to end the chat and delete embeddings from Pinecone.
