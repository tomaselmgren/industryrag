import streamlit as st
from industryrag.industry_rag import DocumentToText, IndustryRAG, DB

st.set_page_config(page_title="Industry RAG Demo")
st.title("Industry RAG Demo")

with st.sidebar:
    db = DB("data.db", "text-mbedding-ada-002")

    with st.spinner('Initializing Database...'):
        db.connect()
        db.create_tables()

    st.header("Model Settings")

    tokens = st.slider("Max Generation Tokens", 10, 1024, 200, 8)

    temperature = st.slider("Temperature", 0.00, 2.00, 0.5)

    model = st.selectbox("Generation Model", ["gpt-3.5-turbo-0125"])

    embeddings_model = st.selectbox("Embeddings Model", ["text-embedding-ada-002"])

    rag = IndustryRAG(db, model_name=model, embeddings_model_name=embeddings_model, max_generation_tokens=tokens, temperature=temperature)

    st.header("Upload Files")

    uploaded_files = st.file_uploader("Choose files", ['docx', 'xls', 'xlsx', 'txt', 'pdf'], True)

    if uploaded_files is not None:

        with st.spinner("Processing Documents..."):
            d2t = DocumentToText()
            d2t.process_and_store_documents(uploaded_files, db)

    st.header("Files Currently in Database")
    
    files = db.get_file_names_in_database()
    if files:
        for file in files:
            st.write(file[0])
    else:
        st.write("no files uploaded yet.")

query = st.text_input("Enter your query: ")

if query:
    with st.spinner("Generating response..."):
        answer = rag.answer(query)
        st.subheader("Response:")
        st.write(answer["answer"])
        st.subheader("Sources:")
        for s in answer["relevant_chunks"]:
            st.write("Document name: " + s["document"] + "\n Similarity: " + str(s["similarity"]))
else:
    st.write()