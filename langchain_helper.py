import os
from langchain_google_genai import GoogleGenerativeAI
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA, HypotheticalDocumentEmbedder
from dotenv import load_dotenv
load_dotenv()

# OpenAI LLM and Embeddings
llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=os.getenv("GOOGLE_API_KEY"))
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Define data source (Lex Fridman transcripts on Kaggle)
transcript_url = "https://www.kaggle.com/datasets/surajsubramanian/lex-fridman-podcast-transcripts"

# Vector database file path
vectordb_file_path = "lex_fridman_podcast_index"

def create_vector_db():
    # Fetch transcripts
    loader = WebBaseLoader(transcript_url)
    transcripts = loader.load()

    # Split into smaller chunks for better context handling
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcripts)

    # Create the vector database
    vectordb = FAISS.from_documents(docs, embeddings)
    vectordb.save_local(vectordb_file_path)


def get_qa_chain():
    # Load the vector database
    vectordb = FAISS.load_local(vectordb_file_path, embeddings)

    # HYDE for enhanced retrieval
    hyde = HypotheticalDocumentEmbedder(llm=llm, base_embeddings=embeddings)
    retriever = vectordb.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5})

    # Updated prompt template for more precise answers
    prompt_template = """Given the following context and a question, generate an answer based on the context. 
                        If the answer is not found in the context, kindly state "I could not find the answer to that in the podcast transcripts."

                        Context: {context}
                        Question: {question}
                        Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    return chain
