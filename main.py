import streamlit as st
from langchain_helper import get_qa_chain, create_vector_db

st.title("Lex Fridman Podcast Q&A")

if st.button("Create Knowledgebase"):
    with st.spinner("Creating knowledgebase, this may take a while..."):
        create_vector_db()

question = st.text_input("Ask me anything about the Lex Fridman Podcast:")

if question:
    chain = get_qa_chain()
    response = chain({"query": question})

    st.subheader("Answer:")
    st.write(response["result"])
