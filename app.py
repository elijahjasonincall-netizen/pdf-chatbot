from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import streamlit as st
import tempfile


st.title("Chat With Your PDF 📄")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(uploaded_file.read())
        temp_path = f.name
    loader = PyPDFLoader(temp_path)
    pages = loader.load()
    # -----splitting----------
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=20)
    chunks = text_splitter.split_documents(pages)
    set_seen = set()
    unique_chunks = []
    for chunk in chunks:
        if chunk.page_content not in set_seen:
            set_seen.add(chunk.page_content)
            unique_chunks.append(chunk)
    # -------embeddings-----
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    # -------vector store-----
    vector_store = Chroma(
        collection_name="pdf_chunks",
        embedding_function=embeddings,
        persist_directory="./chroma_db",
    )
    # After creating vector_store, replace add_documents with this:

    if "vector_store_ready" not in st.session_state:
        vector_store.delete_collection()
        vector_store = Chroma(
            collection_name="pdf_chunks",
            embedding_function=embeddings,
            persist_directory="./chroma_db",
        )
        vector_store.add_documents(unique_chunks)
        st.session_state.vector_store_ready = True
        print("chunks added fresh!")
    else:
        print("using existing chunks!")
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    )
    # -------LLM-----
    model = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key="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",
    )

    def answer_question(question: str) -> str:
        # if asking for summary/explanation get more chunks
        if any(
            word in question.lower()
            for word in ["explain", "summarize", "slide", "all", "everything"]
        ):
            k = 10  # get more chunks
        else:
            k = 3  # normal question

        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": k}
        )
        relevant_chunks = retriever.invoke(question)
        context = "\n".join([chunk.page_content for chunk in relevant_chunks])
        print(len(relevant_chunks))
        rsponse = model.invoke(
            [
                HumanMessage(
                    content=f"""You are a document assistant.
Your ONLY job is to answer based on the EXTRACTED TEXT below.
The text below IS the PDF. It has already been read for you.
Do NOT ask for more context.
Do NOT say you need the PDF.
Just answer directly from the text below.

EXTRACTED PDF TEXT:
{context}

USER QUESTION: {question}

Answer directly and only from the text above.
If information is not in the text, say exactly: "This information is not in the PDF."
Do not make anything up."""
                )
            ]
        )

        return rsponse.content

    # ---ui----

    question = st.text_input("Ask about your PDF:")
    if st.button("Ask"):
        if question:
            with st.spinner("Thinking..."):
                answer = answer_question(question)
            st.markdown(answer)
    else:
        st.info("Please upload a PDF to get started!")
