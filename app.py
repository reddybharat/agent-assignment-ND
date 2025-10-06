import streamlit as st
import os
import tempfile
from src.graphs.builder import build_graph
from src.graphs.type import RAGAgentState
from src.utils.ingest_pdf import IngestPDF

# Configure page
st.set_page_config(
    page_title="AI Assistant",
    layout="centered"
)

# Initialize session state
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'answer' not in st.session_state:
    st.session_state.answer = ""
if 'ingestion_completed' not in st.session_state:
    st.session_state.ingestion_completed = False

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


def upload_and_ingest_tab():
    """Tab for uploading files and running ingestion"""
    st.subheader("Upload and Ingest Files")
    
    # File upload section
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        accept_multiple_files=True,
        type=['pdf']
    )
    
    if uploaded_files:
        st.write(f"Uploaded {len(uploaded_files)} file(s)")
        for file in uploaded_files:
            st.write(f"- {file.name}")
    
    # Ingest button
    if st.button("Ingest Files", type="primary", disabled=not uploaded_files):
        if uploaded_files:
            with st.spinner("Ingesting files..."):
                try:
                    # Save uploaded files temporarily
                    file_paths = save_uploaded_files(uploaded_files)
                    
                    # Run ingestion pipeline
                    ingestor = IngestPDF()
                    ingestor.run_ingestion_pipeline(file_paths)
                    
                    # Mark ingestion as completed
                    st.session_state.ingestion_completed = True
                    st.session_state.uploaded_files = file_paths
                    
                    st.success("Files ingested successfully! You can now ask questions in the 'Ask Questions' tab.")
                    
                    # Clean up temporary files
                    for file_path in file_paths:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                            
                except Exception as e:
                    st.error(f"Error during ingestion: {str(e)}")
        else:
            st.warning("Please upload files first.")
    
    # Show ingestion status
    if st.session_state.ingestion_completed:
        st.success("Files have been ingested and are ready for querying!")

def ask_questions_tab():
    """Tab for asking questions"""
    st.subheader("Ask Questions")
    
    # Check if ingestion has been completed
    if not st.session_state.ingestion_completed:
        st.warning("Please upload and ingest files first in the 'Upload Files' tab.")
        return
    
    # Query input
    user_query = st.text_area(
        "Enter your query:",
        placeholder="Ask anything about your uploaded documents...",
        height=100
    )
    
    # Submit button
    if st.button("Submit Query", type="primary"):
        if user_query.strip():
            with st.spinner("Processing your query..."):
                try:
                    # Create initial state for the graph
                    initial_state = RAGAgentState(
                        files_uploaded=[],  # No files needed since they're already ingested
                        query=user_query,
                        answer="",
                        status="processing",
                        is_weather_query=False,
                        location=""
                    )
                    
                    # Build and run graph
                    graph = build_graph(initial_state)
                    final_state = graph.invoke(initial_state)
                    
                    # Update session state with the answer
                    st.session_state.answer = final_state['answer']
                    
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
        else:
            st.warning("Please enter a query.")
    
    # Display result
    if st.session_state.answer:
        st.subheader("Response")
        st.write(st.session_state.answer)

def main():
    # Center the content
    col1, col2, col3 = st.columns([1, 6, 1])
    
    with col2:
        st.title("AI Assistant")
        
        # Create tabs
        tab1, tab2 = st.tabs(["Upload Files", "Ask Questions"])
        
        with tab1:
            upload_and_ingest_tab()
        
        with tab2:
            ask_questions_tab()

if __name__ == "__main__":
    main()
