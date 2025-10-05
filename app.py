import streamlit as st
import os
import tempfile
from src.graphs.builder import build_graph
from src.graphs.type import RAGAgentState

# Configure page
st.set_page_config(
    page_title="AI Assistant",
    page_icon="🤖",
    layout="centered"
)

# Initialize session state
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'answer' not in st.session_state:
    st.session_state.answer = ""

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
    # Center the content
    col1, col2, col3 = st.columns([1, 6, 1])
    
    with col2:
        st.title("AI Assistant")
        
        # File upload section
        st.subheader("Upload Files")
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            accept_multiple_files=True
        )
        
        # Query input
        st.subheader("Ask a Question")
        user_query = st.text_area(
            "Enter your query:",
            placeholder="Ask anything...",
            height=100
        )
        
        # Submit button
        if st.button("Submit", type="primary"):
            # Clear previous answer for fresh start
            st.session_state.answer = ""
            
            if uploaded_files and user_query.strip():
                with st.spinner("Processing your request..."):
                    try:
                        # Save uploaded files temporarily
                        file_paths = save_uploaded_files(uploaded_files)
                        
                        # Create fresh initial state
                        initial_state = RAGAgentState(
                            files_uploaded=file_paths,
                            query=user_query,
                            answer="",
                            status="processing",
                            is_weather_query=False,
                            location=""
                        )
                        
                        # Build and run fresh graph
                        graph = build_graph(initial_state)
                        final_state = graph.invoke(initial_state)
                        
                        # Update session state with the answer
                        st.session_state.answer = final_state['answer']
                        
                        # Clean up temporary files
                        for file_path in file_paths:
                            if os.path.exists(file_path):
                                os.remove(file_path)
                                
                    except Exception as e:
                        st.error(f"Error processing request: {str(e)}")
            else:
                st.warning("Please upload files and enter a query.")
        
        # Display result
        if st.session_state.answer:
            st.subheader("Response")
            st.write(st.session_state.answer)

if __name__ == "__main__":
    main()
