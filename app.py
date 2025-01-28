# application.py

import streamlit as st
import PyPDF2
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Function to extract text from PDF (updated to work with file-like objects)
def extract_text_from_pdf(pdf_file):  # Correct parameter name: pdf_file
    reader = PyPDF2.PdfReader(pdf_file)  # Use the correct parameter name
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to chunk text
def chunk_text(text, chunk_size=512):
    words = text.split()
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Function to extract relevant text
def extract_relevant_text(prompt, text_chunk):
    input_text = f"{prompt}\n\n{text_chunk}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Generate output
    output = model.generate(input_ids, max_length=1024, num_return_sequences=1)
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return output_text

# Function to process articles
def process_articles(articles):
    results = []
    for article_id, article_text in enumerate(articles):
        chunks = chunk_text(article_text)
        for chunk_id, chunk in enumerate(chunks):
            prompt = "Extract relevant text for a literature review:"
            extracted_text = extract_relevant_text(prompt, chunk)
            results.append({
                "extracted_text": extracted_text,
                "source_article": f"Article {article_id + 1}",
                "chunk_id": chunk_id
            })
    return results

# Streamlit app
def main():
    st.title("Literature Review Text Extractor")

    # File uploader for multiple PDFs
    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        articles = []
        for uploaded_file in uploaded_files:
            # Extract text from each uploaded PDF
            text = extract_text_from_pdf(uploaded_file)  # Pass the file-like object directly
            articles.append(text)

        # Process the articles
        results = process_articles(articles)

        # Display results
        st.header("Extracted Text")
        for result in results:
            st.subheader(f"Source: {result['source_article']}, Chunk: {result['chunk_id']}")
            st.write(result['extracted_text'])
            st.write("-" * 50)

if __name__ == "__main__":
    main()
