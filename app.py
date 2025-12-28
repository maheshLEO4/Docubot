import streamlit as st
from auth import setup_authentication
from config import validate_config
from query_processor import process_query
from data_processing import save_uploaded_files, get_document_chunks
from vector_store import build_vector_store_from_pdfs

st.set_page_config(page_title="DocuBot", page_icon="⚡", layout="wide")

def main():
    if not validate_config(): return
    
    user_id = setup_authentication()
    if not user_id: return

    st.title("⚡ DocuBot Pro")

    # Sidebar: Knowledge Base Management
    with st.sidebar:
        st.header("📁 Knowledge Base")
        files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
        if st.button("Index Documents", type="primary", use_container_width=True):
            if files:
                with st.spinner("Processing..."):
                    paths = save_uploaded_files(files, user_id)
                    build_vector_store_from_pdfs(user_id, paths)
                    st.success("Indexing Complete!")
            else:
                st.warning("Please upload files first.")

    # Chat Interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = process_query(user_id, prompt)
            st.markdown(response['answer'])
            st.session_state.messages.append({"role": "assistant", "content": response['answer']})

if __name__ == "__main__":
    main()