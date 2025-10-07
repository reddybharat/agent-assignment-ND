# AI Assistant

An application that combines document processing with weather information services. Built using LangGraph for workflow orchestration, this application can process uploaded PDF documents and answer questions about their content, or provide real-time weather information.

## Table of Contents

- [Project Structure](#project-structure)
- [Project Architecture](#project-architecture)
  - [Document Ingestion Pipeline](#document-ingestion-pipeline)
  - [Graph Structure](#graph-structure)
- [Node Definitions](#node-definitions)
  - [Routing Node](#1-routing-node-srcgraphsnodesrouting_nodepy)
  - [Weather Node](#2-weather-node-srcgraphsnodesweather_nodepy)
  - [Retriever Node](#3-retriever-node-srcgraphsnodesretriever_nodepy)
- [State Management](#state-management)
- [Environment Setup](#environment-setup)
- [Local Setup and Running](#local-setup-and-running)
  - [Prerequisites](#prerequisites)
  - [Installation Steps](#installation-steps)
- [Workflow Process](#workflow-process)
  - [How to Use the Application](#how-to-use-the-application)
  - [Document Ingestion Workflow](#document-ingestion-workflow)
  - [Query Processing Workflow](#query-processing-workflow)
- [Screenshots](#screenshots)
- [Testing](#testing)
- [LangSmith Evaluation](#langsmith-evaluation)
- [Error Handling](#error-handling)
- [Dependencies](#dependencies)
- [Usage Examples](#usage-examples)

## Project Structure

```
agent-assignment-ND/
├── app.py                          # Streamlit application entry point
├── requirements.txt                # Python dependencies
├── .env                           # Environment variables (create this)
├── src/
│   ├── graphs/
│   │   ├── builder.py             # Graph construction and routing logic
│   │   ├── type.py                # State type definitions
│   │   └── nodes/
│   │       ├── routing_node.py    # Query classification
│   │       ├── weather_node.py    # Weather information retrieval
│   │       └── retriever_node.py  # Document-based Q&A
│   └── utils/
│       ├── ingest_pdf_docling_genaiembeddings.py  # Current PDF processing (Docling + Gemini)
│       ├── ingest_pdf.py          # Legacy PDF processing (unused)
│       ├── ingest_pdf_docling.py   # Legacy PDF processing (unused)
│       ├── openweather.py         # Weather API integration
│       ├── prompts.py             # LLM prompt templates
│       └── retriever.py           # Vector search and RAG implementation
├── evaluation/
│   └── langsmith_evaluator.py     # LangSmith evaluation utility (optional)
├── tests/
│   ├── run_tests.py               # Test runner script
│   ├── test_retriever.py          # Document retrieval tests
│   ├── test_llm_processing.py     # LLM processing and routing tests
│   └── test_weather_api.py        # Weather API integration tests
└── README.md                      # This file
```

***Note**: The `ingest_pdf.py` and `ingest_pdf_docling.py` files are legacy and unused. The current implementation uses `ingest_pdf_docling_genaiembeddings.py`.*

## Project Architecture

This project implements a comprehensive RAG system with two main components: a document ingestion pipeline and a state-based graph architecture using LangGraph. The system intelligently routes queries between document retrieval and weather services based on the user's intent.

### Document Ingestion Pipeline
**Purpose**: Processes and indexes uploaded PDF documents into a vector database using advanced document parsing.

***Note** : This process occurs separately from query processing and must be completed before documents can be queried.*


#### Ingestion Architecture

```
                    ┌─────────────────┐
                    │   PDF Upload    │
                    │  (Streamlit)    │
                    └─────────┬───────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Document Parse │
                    │    (Docling)    │
                    └─────────┬───────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Text Chunking  │
                    │ (Markdown + RC) │
                    └─────────┬───────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │    Embedding    │
                    │   (Gemini 001)  │
                    └─────────┬───────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Vector Storage │
                    │  (Qdrant Cloud) │
                    └─────────────────┘
```

**Ingestion Process Flow**: (`src/utils/ingest_pdf_docling_genaiembeddings.py`)
1. **File Upload**: Users upload PDF files through Streamlit interface
2. **Document Parsing**: Docling extracts text and structure from PDFs
3. **Text Chunking**: Chunking based based on Markdown headers and recursive character.
4. **Embedding Generation**: Google Gemini creates vector embeddings
5. **Database Storage**: Chunks stored in Qdrant with metadata

### Graph Structure

```
                    ┌─────────────────┐
                    │      START      │
                    └─────────┬───────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Routing Node   │
                    │ (Query Analysis)│
                    └─────────┬───────┘
                              │
                              ▼
                    ┌─────────┴───────┐
                    │                 │
                    ▼                 ▼
            ┌─────────────┐   ┌─────────────┐
            │   Weather   │   │  Retriever  │
            │    Node     │   │    Node     │
            │(Weather API)│   │   (RAG)     │
            └─────────────┘   └─────────────┘
                    │                 │
                    └─────────┬───────┘
                              │
                              ▼
                             END
```

**Detailed Flow**:
1. **START**: Initial state with user query
2. **Routing Node**: Analyzes query to determine if it's weather-related. Routes to either Weather or Retriever node
3. **Weather Node**: Handles weather queries using OpenWeatherMap API
4. **Retriever Node**: Handles document-based queries using RAG
5. **END**: Returns final answer to user

## Node Definitions

### 1. Routing Node (`src/graphs/nodes/routing_node.py`)
**Purpose**: User query classification to route to appropriate node.

**Functionality**:
- Uses Google Gemini to analyze query intent
- Analyzes query for weather-related keywords and extracts location information from weather-related queries
- Returns boolean flag and location string

### 2. Weather Node (`src/graphs/nodes/weather_node.py`)
**Purpose**: Provides real-time weather information for specified location.

**Functionality**:
- Geocodes location names to coordinates using OpenWeatherMap Geocoding API
- Retrieves current weather data using coordinates
- Formats weather information (current temperature and weather conditions) for better readability


### 3. Retriever Node (`src/graphs/nodes/retriever_node.py`)
**Purpose**: Performs document-based question answering using RAG methodology.

**Functionality**:
- Performs similarity search on vector database (QdrantDB)
- Retrieves top-k most relevant document chunks
- Generates comprehensive answers using Google Gemini.


## State Management

The application uses a TypedDict to manage state throughout the graph execution:

```python
class RAGAgentState(TypedDict):
    query: str          # User's input query
    answer: str         # Generated response
    status: str         # Current processing status
    is_weather_query: bool  # Whether query is weather-related
    location: str       # Extracted location for weather queries
```

**State Flow**:
1. **Initial**: Query received, status set to "processing"
2. **Routing**: Query analyzed, `is_weather_query` and `location` set
3. **Processing**: Appropriate node (weather/retriever) processes query
4. **Completed**: Final answer stored in `answer` field, status updated

## Environment Setup

Create a `.env` file in the project root with the following variables:

```bash
# Google Gemini API Key (Required)
GOOGLE_API_KEY=your_google_api_key_here

# OpenWeatherMap API Key (Required for weather functionality)
OPENWEATHER_API_KEY=your_openweather_api_key_here

# Qdrant Cloud Configuration (Required for vector storage)
QDRANT_CLOUD_URL=https://your-cluster-url.qdrant.tech
QDRANT_API_KEY=your_qdrant_api_key_here

# LangSmith Configuration (Optional - for evaluation and tracing)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=rag-agent-evaluation
```

### API Key Setup Instructions

1. **Google API Key**:
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Copy the key to your `.env` file

2. **OpenWeatherMap API Key**:
   - Sign up at [OpenWeatherMap](https://openweathermap.org/api)
   - Generate a free API key
   - Copy the key to your `.env` file

3. **Qdrant Cloud Setup**:
   - Create account at [Qdrant Cloud](https://cloud.qdrant.io/)
   - Create a new cluster
   - Copy the cluster URL and API key to your `.env` file

4. **LangSmith Setup** (Optional):
   - Sign up at [LangSmith](https://smith.langchain.com/)
   - Create a new project called "rag-agent-evaluation"
   - Generate an API key from your account settings
   - Copy the API key to your `.env` file
   - **Tracing**: Automatically traces all LLM calls and graph executions
   - **Evaluation**: Run model evaluations and view results on the web dashboard

## Local Setup and Running

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd agent-assignment-ND
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   - Create `.env` file in project root
   - Add all required API keys (see Environment Setup section)

5. **Run the application**:
   ```bash
   streamlit run app.py
   ```

6. **Access the application**:
   - Open your browser to `http://localhost:8501`

## Workflow Process

### How to Use the Application

Once you have the application running (`streamlit run app.py`), follow these steps:

#### 1. Document Ingestion (Required First Step)
- **Upload PDF Files**: Use the file uploader in the Streamlit interface to upload your PDF documents
- **Process Documents**: Click the "Process Documents" button to ingest them into the vector database
- **Wait for Completion**: The system will parse, chunk, and embed your documents (this may take a few minutes)
- **Confirmation**: You'll see a success message when documents are ready for querying

#### 2. Query Processing
- **Ask Questions**: Use the chat interface to ask questions about your uploaded documents
- **Weather Queries**: Ask about weather in any location (e.g., "What's the weather in New York?")
- **Document Queries**: Ask questions about your uploaded documents (e.g., "What are the main findings?")

#### 3. Understanding Responses
- **Weather Responses**: Get current temperature, conditions, and location-specific weather data
- **Document Responses**: Get answers based on the content of your uploaded PDFs
- **Routing**: The system automatically determines whether your query is about weather or documents

### Document Ingestion Workflow
1. **File Upload**: User uploads PDF files through the Streamlit interface
2. **Document Parsing**: Docling extracts text and structure from PDFs
3. **Text Chunking**: Markdown headers and recursive character splitting
4. **Embedding Generation**: Google Gemini creates vector embeddings
5. **Database Storage**: Chunks stored in Qdrant with metadata

### Query Processing Workflow
1. **Query Input**: User submits a natural language question
2. **Routing**: System determines if query is weather-related or document-based
3. **Processing**: Appropriate node (Weather or Retriever) processes the query
4. **Response**: Generated answer is displayed to the user

***Note**: Document ingestion and query processing are separate workflows. Documents must be ingested before queries can be processed.*

## Testing

The project includes a comprehensive test suite covering all major components with simple, focused tests that verify core functionality.

### Running Tests

**Quick Start**:
```bash
# Run all tests (recommended)
python -m pytest tests/ -v

# Or use the custom test runner
python -m tests.run_tests

# Run specific test file
python -m pytest tests/test_retriever.py -v

# Run specific test function
python -m pytest tests/test_retriever.py::test_retriever_init_success -v
```

**Important**: Always use `python -m pytest` instead of just `pytest` to avoid import path issues across different systems.

**Visual Example**: See the [Screenshots](#screenshots) section for a visual example of the test execution results.

### Test Structure

The test suite includes three main test files:

#### 1. `test_llm_processing.py` - LLM Processing and State Management Tests
Tests the core LLM processing, routing logic, and state management:
- **Routing Condition Tests**: Verifies that queries are correctly routed to weather or retriever nodes
- **State Management Tests**: Tests state field types, transitions, and modifications
- **Query Classification Tests**: Validates different query types (weather vs document queries)
- **State Status Transitions**: Tests various status states (initial, routing, weather, retriever, completed, error)
- **Location Handling**: Tests location field handling for weather queries
- **Answer Processing**: Tests answer field handling and modification
- **State Field Validation**: Tests required fields and field modifications


#### 2. `test_retriever.py` - Document Retrieval Tests
Tests the RAG document retrieval functionality:
- **Initialization Tests**: Verifies retriever setup with and without API keys
- **Document Retrieval**: Tests retrieval with different queries and k values
- **Response Generation**: Tests LLM response generation for various queries
- **Custom Parameters**: Tests retrieval with different k values (top-k results)


#### 3. `test_weather_api.py` - Weather API Tests
Tests the OpenWeatherMap API integration:
- **Geocoding Tests**: Tests location geocoding for major cities (London, New York, Tokyo, Paris, Sydney)
- **Invalid Location Handling**: Tests error handling for non-existent locations
- **Weather Data Retrieval**: Tests weather data fetching for valid coordinates
- **Data Structure Validation**: Verifies all required weather fields are present
- **Error Handling**: Tests API error responses for invalid coordinates


### Prerequisites for Testing

```bash
pip install pytest
```
***Note** : Should already be installed with requirements.txt*

**Environment Requirements**:
- Valid API keys in `.env` file (Google API, OpenWeatherMap, Qdrant)
- Internet connection for API testing
- Python 3.8 or higher


## LangSmith Evaluation

The project includes a separate LangSmith evaluation utility for testing the RAG agent's performance. This is located in `evaluation/langsmith_evaluator.py` and is **not part of the main application**.

### Purpose

The LangSmith evaluator is a standalone utility for:
- Testing the RAG agent's performance with predefined datasets
- Evaluating routing accuracy and response quality
- Running comprehensive model evaluations

### Usage

***Note**: This is a separate evaluation tool, not integrated into the main application.*

```bash
# Run the evaluation script directly
python -m evaluation.langsmith_evaluator
```

### What It Tests

The evaluation tests:

1. **Routing Accuracy**: Are weather queries correctly identified as weather queries?
2. **Response Quality**: Does the agent provide accurate answers to queries?
3. **Graph Performance**: Tests the complete graph execution flow
4. **Retrieval Performance**: Tests document retrieval and RAG functionality

### Evaluation Process

1. **Dataset Creation**: Creates a custom dataset with predefined test cases
2. **Test File Ingestion**: Ingests test documents into the vector database for evaluation
3. **Retriever Evaluation**: Tests the document retrieval component separately
4. **Graph Evaluation**: Tests the complete LangGraph workflow end-to-end
5. **Results Analysis**: View detailed results and metrics on LangSmith web UI

### Evaluation Features

- **Custom Dataset**: Predefined test cases for consistent evaluation
- **Test Document Ingestion**: Automatically ingests test files into vector database
- **Similarity Evaluation**: Uses Gemini to evaluate response quality
- **Graph Evaluation**: Tests the complete LangGraph workflow
- **Retrieval Evaluation**: Tests document retrieval separately
- **Async Processing**: Handles multiple evaluations concurrently
- **LangSmith Integration**: Results viewable on LangSmith web dashboard

### Tracing and Monitoring

With LangSmith tracing enabled in the environment variables, you can monitor and analyze:

- **Request Tracing**: View detailed traces for every API call and LLM interaction
- **Performance Metrics**: Track response times, token usage, and costs
- **Error Analysis**: Identify and debug issues in the workflow
- **Graph Execution**: Visualize the complete graph execution flow
- **Retrieval Analysis**: Monitor document retrieval performance and relevance

**Tracing Features**:
- Real-time monitoring of all LLM calls
- Detailed execution traces for each graph node
- Performance metrics and timing analysis
- Error tracking and debugging information
- Cost analysis for API usage

**Visual Examples**: See the [Screenshots](#screenshots) section for visual examples of:
- LangGraph execution traces for RAG queries
- LangGraph execution traces for weather queries
- Custom dataset creation and evaluation results

### Prerequisites for Evaluation

- Valid LangSmith API key in `.env` file
- All other API keys (Google, OpenWeatherMap, Qdrant)
- Internet connection for API calls
- LangSmith tracing enabled (`LANGCHAIN_TRACING_V2=true`)

***Note**: The evaluator is optional and only needed for model evaluation and testing. It does not affect the main application functionality.*


## Error Handling

The application includes comprehensive error handling:

- **File Processing**: Graceful handling of corrupted or empty PDFs
- **API Failures**: Fallback responses for external API failures
- **Network Issues**: Timeout handling for API requests
- **Invalid Queries**: Clear error messages for malformed requests

## Dependencies

### Core Dependencies
- **LangGraph**: Workflow orchestration and state management
- **LangChain**: LLM integration and document processing
- **Streamlit**: Web application framework
- **Qdrant**: Vector database for document storage
- **Docling**: Advanced PDF parsing and text extraction
- **Google Gemini**: Text generation and embedding models

### API Integrations
- **Google Gemini**: Large language model for text generation and embeddings
- **OpenWeatherMap**: Weather data API
- **Qdrant Cloud**: Vector database hosting
- **LangSmith**: Optional evaluation and monitoring (evaluation only)

## Usage Examples

### Document Queries
```
"What are the main findings in the research paper?"
"Summarize the key points from the uploaded document"
"What methodology was used in this study?"
```

### Weather Queries
```
"What's the weather in New York?"
"How's the temperature in London today?"
"Is it raining in Tokyo?"
```