import streamlit as st
from rag_pipeline import process_url_and_query

# Streamlit App
st.title("Chat with Websites using RAG Pipeline")
st.write("Crawl, Scrape, and Ask Questions on Websites!")

# Input Section
url = st.text_input("Enter the website URL (e.g., https://www.stanford.edu/):")
query = st.text_input("Enter your question:")

# Submit button
if st.button("Submit"):
    if url and query:
        with st.spinner("Processing..."):
            response = process_url_and_query(url, query)
        st.subheader("Response:")
        st.write(response)
    else:
        st.warning("Please provide both a URL and a question.")
