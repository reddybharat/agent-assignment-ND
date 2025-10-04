import streamlit as st
import os
import tempfile
from src.utils.ingest_pdf import IngestPDF
from src.utils.retriever import Retriever

# Configure page
st.set_page_config(
    page_title="PDF Chat Assistant",
    page_icon="📄",
    layout="wide"
)

# Initialize session state
if 'vector_store_ready' not in st.session_state:
    st.session_state.vector_store_ready = False
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'retriever' not in st.session_state:
    st.session_state.retriever = None

def save_uploaded_files(uploaded_files):
    """Save uploaded files to temporary directory and return file paths"""
    temp_dir = tempfile.mkdtemp()
    file_paths = []
    
    for uploaded_file in uploaded_files:
        # Create a temporary file
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
        
        # Write the uploaded file to the temporary file
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        file_paths.append(temp_file_path)
    
    return file_paths

def main():
    st.title("📄 PDF Chat Assistant")
    st.markdown("Upload PDF files, ingest them into a vector database, and ask questions about their content.")
    
    # Upload & Ingest Section
    st.header("📁 Upload & Ingest PDFs")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload one or more PDF files to analyze"
    )
    
    
    # Ingest button
    if uploaded_files and st.button("🚀 Ingest PDFs", type="primary"):
        with st.spinner("Processing PDFs and creating vector embeddings..."):
            try:
                # Save uploaded files temporarily
                file_paths = save_uploaded_files(uploaded_files)
                
                # Initialize ingestor
                ingestor = IngestPDF()
                
                # Ingest PDFs
                vector_store = ingestor.run_ingestion_pipeline(file_paths)
                
                # Update session state
                st.session_state.vector_store_ready = True
                st.session_state.uploaded_files = [f.name for f in uploaded_files]
                # Note: Retriever will be initialized when user asks first question
                
                if vector_store is not None:
                    st.success("✅ PDFs successfully ingested! You can now ask questions.")
                else:
                    st.warning("⚠️ PDFs ingested with warnings. Some features may be limited, but you can still try asking questions.")
                
                # Clean up temporary files
                for file_path in file_paths:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                
            except Exception as e:
                st.error(f"❌ Error ingesting PDFs: {str(e)}")
    
    # Separator
    st.divider()
    
    # Chat Section
    st.header("💬 Chat with Your PDFs")
    
    # Check if vector store is ready
    if not st.session_state.vector_store_ready:
        st.info("👆 Please upload and ingest PDF files first to start chatting.")
    else:
        st.success(f"✅ Ready to chat! Ingested {len(st.session_state.uploaded_files)} file(s)")
        
        
        # Query input
        user_query = st.text_area(
            "Ask a question about your PDFs:",
            placeholder="e.g., What is the main topic discussed in the documents?",
            height=100,
            help="Enter your question about the uploaded PDF content"
        )
        
        # Chat button
        if st.button("💭 Generate Response", type="primary") and user_query.strip():
            with st.spinner("Searching documents and generating response..."):
                try:
                    # Initialize retriever only when needed
                    if st.session_state.retriever is None:
                        st.session_state.retriever = Retriever()
                    
                    # Generate response using retriever
                    response = st.session_state.retriever.generate_response(user_query)
                    
                    # Display response
                    st.markdown("### 🤖 Response:")
                    st.markdown(response)
                    
                except Exception as e:
                    st.error(f"❌ Error generating response: {str(e)}")
        elif user_query.strip() == "":
            st.warning("⚠️ Please enter a question before clicking 'Generate Response'.")

if __name__ == "__main__":
    main()
