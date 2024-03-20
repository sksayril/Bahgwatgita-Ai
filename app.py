import streamlit as st
import langchain
from langchain import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import textwrap
import warnings
from langchain.llms import HuggingFacePipeline
import os

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

warnings.filterwarnings("ignore")

DB_FAISS_PATH = 'vectorstore/db_faiss'  # Replace with your actual path

# Load your existing functions

def wrap_text_preserve_newlines(text, width=200):
    lines = text.split('\n')
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    wrapped_text = '\n'.join(wrapped_lines)
    return wrapped_text

def process_llm_response(llm_response):
    ans = wrap_text_preserve_newlines(llm_response['result'])
    sources_used = '\n'.join([str(source.metadata['source']) for source in llm_response['source_documents']])
    ans = ans + '\n\nSources:\n' + sources_used
    return ans

def set_custom_prompt():
    custom_prompt_template = """Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    Context: {context}
    Question: {question}
    
    Only return the helpful answer below and nothing else.
    Helpful answer:
    """
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
    return prompt

def load_llm():
    llm = CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm

def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = RetrievalQA.from_chain_type(llm=llm,
                                     chain_type='stuff',
                                     retriever=db.as_retriever(search_kwargs={'k': 2}),
                                     return_source_documents=True,
                                     chain_type_kwargs={'prompt': qa_prompt})
    return qa

def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response

def main():
    st.title("Bhagavad Gita AI")
    st.write("This AI content does not necessarily have to believe all responses; it has been trained on open-source datasets.")
    # User input for the query
    query = st.text_area("Enter your question about Bhagavad Gita:")

    if st.button("Get Answer"):
        # Get the response using your existing function
        result = final_result(query)

        # Process the response for display
        formatted_response = process_llm_response(result)

        # Display the response
        st.markdown(f"**Answer:**\n\n{formatted_response}")

if __name__ == "__main__":
    main()
