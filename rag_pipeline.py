import os
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import streamlit as st

# Set your Hugging Face API key
API_KEY = "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"

# Define Hugging Face API URL for question answering model
API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-distilled-squad"

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# 1. Function to crawl and scrape websites
def scrape_website(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()
        return text
    except Exception as e:
        return f"Error: {str(e)}"

# 2. Chunk and embed the scraped content
def embed_content(text):
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    
    # Generate embeddings
    embeddings = embedding_model.encode(chunks)
    return chunks, np.array(embeddings).astype('float32')

# 3. Store embeddings in FAISS
def store_in_faiss(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# 4. Query handling with Hugging Face API
def query_faiss(index, chunks, query, k=3):
    # Embed the query
    query_embedding = embedding_model.encode([query]).astype('float32')
    
    # Search in FAISS index
    distances, indices = index.search(query_embedding, k)
    results = [chunks[i] for i in indices[0]]
    return results

# 5. Generate response using Hugging Face
# 5. Generate response using Hugging Face
def generate_response(retrieved_text, query):
    prompt = f"Answer the question: '{query}' using the following context:\n\n{retrieved_text}"
    
    headers = {"Authorization": f"Bearer {API_KEY}"}
    data = {
        "inputs": {
            "question": query,
            "context": retrieved_text
        }
    }

    # Send request to Hugging Face API
    response = requests.post(API_URL, headers=headers, json=data)
    
    if response.status_code == 200:
        # Extract the answer from the response
        response_json = response.json()
        print("Response JSON:", response_json)  # Debugging line to inspect the structure
        
        # Get the answer from the response
        answer = response_json.get('answer', 'No answer found')  # Use get() to safely access 'answer'
        return answer
    else:
        return f"Error: {response.status_code}, {response.text}"


# Main pipeline
def process_url_and_query(url, query):
    text = scrape_website(url)
    if "Error" in text:
        return text
    chunks, embeddings = embed_content(text)
    index = store_in_faiss(embeddings)
    retrieved_texts = query_faiss(index, chunks, query)
    final_response = generate_response("\n".join(retrieved_texts), query)
    return final_response

# Streamlit UI to interact with the pipeline
def main():
    st.title("Website Q&A with RAG Pipeline")

    # User inputs
    url = st.text_input("Enter the URL of the webpage:")
    query = st.text_input("Ask a question based on the webpage:")

    if url and query:
        st.write("Processing your request...")

        # Process the URL and query
        result = process_url_and_query(url, query)

        # Display the result
        st.write("Answer:")
        st.write(result)

if __name__ == "__main__":
    main()
