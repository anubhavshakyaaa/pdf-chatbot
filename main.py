import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from groq import Groq
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

# -------------------- Load environment variables --------------------
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# -------------------- Initialize Pinecone --------------------
pinecone_client = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

# Create index if it does not exist
if PINECONE_INDEX_NAME not in pinecone_client.list_indexes().names():
    pinecone_client.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=384,  # embedding dimension for all-MiniLM-L6-v2
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENVIRONMENT),
    )

index = pinecone_client.Index(PINECONE_INDEX_NAME)

# -------------------- Initialize Groq --------------------
groq_client = Groq(api_key=GROQ_API_KEY)

# -------------------- Initialize embeddings & text splitter --------------------
embedder = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)

# -------------------- PDF Text Extraction --------------------
def extract_text_from_pdfs(pdf_paths):
    all_text = ""
    for path in pdf_paths:
        pdf = PdfReader(path)
        for page in pdf.pages:
            all_text += page.extract_text() or ""
    return all_text

# -------------------- Text Chunking --------------------
def chunk_text(text, chunk_size=800, overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)

# -------------------- Create & Upload Embeddings --------------------
def vec_to_pinecone(chunks):
    pinecone_data = []
    for i, chunk in enumerate(tqdm(chunks, desc="Creating embeddings")):
        embedding = embedder.embed_query(chunk)
        metadata = {"chunk_id": i, "text": chunk}
        pinecone_data.append((f"chunk_{i}", embedding, metadata))

    index.upsert(vectors=pinecone_data)
    print("✅ Embeddings successfully stored in Pinecone!")

# -------------------- Retrieve Context from Pinecone --------------------
def retrieve_context(query):
    query_vector = embedder.embed_query(query)
    search_results = index.query(vector=query_vector, top_k=8, include_metadata=True)

    print(f"🔎 Retrieved {len(search_results['matches'])} relevant chunks from Pinecone.")
    contexts = [result["metadata"]["text"] for result in search_results["matches"]]
    return "\n".join(contexts)

# -------------------- Generate Answer Using Groq --------------------
def generate_answer_with_groq(query, context):
    system_prompt = f"""
You are a factual and context-restricted assistant.

You are given some information in the CONTEXT section below. Analyse and understand it carefully. 
Your task is to answer the user's question strictly based on the information in the CONTEXT only.

Guidelines:
- Use only the information provided in CONTEXT.
- Do NOT use any external knowledge, assumptions, or general facts.
- If the answer is not explicitly available in the CONTEXT, respond with:
"I don’t have enough information in the provided context to answer that."
- Do not generate or guess missing information.
- Always keep your answers in detail, and directly relevant to the question.
- If multiple pieces of information are relevant, synthesize them into a coherent answer.
- Strictly do not include your thinking process in the final answer.
"""
    prompt = f"{system_prompt}\n\nContext:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # update if model is deprecated
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()

# -------------------- Main Program --------------------
if __name__ == "__main__":
    # Provide your PDF files here
    pdf_files = ["files/test_file.pdf"]  # replace with your file paths
    text = extract_text_from_pdfs(pdf_files)
    chunks = chunk_text(text)
    vec_to_pinecone(chunks)

    while True:
        user_query = input("\n❓ Ask a question about your PDFs (or type 'exit'): ")
        if user_query.lower() == "exit":
            index.delete(delete_all=True, namespace='__default__')
            break
        context = retrieve_context(user_query)
        answer = generate_answer_with_groq(user_query, context)

        print("💬 Answer:\n", answer)
