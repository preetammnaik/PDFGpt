import os.path
import os

import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

with st.sidebar:
    st.title('PDF GPT')
    st.markdown('''
     ## About
     This app is an LLM-powered chatbot built using:
     - [Streamlit](https://streamlit.io/)
     - [LangChain](https://python.langchain.com/)
     - [OpenAI](https://platform.openai.com/docs/models) LLM model
     
     Here one can upload any 'PDF' files and ask any question realted to
     the PDF, and the app tries to answer the users query.
     ''')
    add_vertical_space(5)


def main():
    st.header('Chat with your PDF')
    load_dotenv()

    # upload PDF
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Will split the text with a size of 1000 and overlap at 200
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        pdf_name = pdf.name[:-4]
        # Now creating embeddings for each chunk for later comparison for similarity

        # check if the filename already exists, read from the disk else compute embeddings using OpenAi
        if os.path.exists(f"{pdf_name}.pkl"):
            with open(f"{pdf_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
        else:
            embeddings = OpenAIEmbeddings()
            if embeddings:
                VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
                with open(f"{pdf_name}.pkl", "wb") as f:
                    pickle.dump(VectorStore, f)
            else:
                st.write("Error: No embeddings were generated. Check your OpenAIEmbeddings setup.")

        # User Query

        query = st.text_input("Ask questions about your PDF File:")
        # finding the similar docs from the knowledge base. returns the top 3 documents
        if query:
            docs = VectorStore.similarity_search(query=query, k=3)

            # Using LLM to find the answers and using gpt 3.5 turbo
            llm = OpenAI(model_name='gpt-3.5-turbo')
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=query)
            st.write(response)


if __name__ == '__main__':
    main()
