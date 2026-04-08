# Retrieval-Augmented Generation (RAG) with Weaviate and LlamaIndex
## Complete Step-by-Step Guide

---

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Code Breakdown](#code-breakdown)
4. [Detailed Explanations](#detailed-explanations)
5. [How It Works](#how-it-works)
6. [Common Issues & Solutions](#common-issues--solutions)

---

## Overview

This code implements a **Retrieval-Augmented Generation (RAG)** system that:
- **Loads** documents from local files
- **Converts** them into embeddings (numerical vectors)
- **Stores** them in a Weaviate vector database
- **Retrieves** relevant documents based on semantic similarity
- **Uses** an LLM (Groq) to answer questions based on the retrieved documents

### Why RAG?
- Allows LLMs to answer questions about **custom/private data**
- Reduces hallucinations by grounding answers in actual documents
- Keeps data **secure** (stays in your database, not sent to the LLM)

---

## Prerequisites

### Required Libraries
```bash
pip install weaviate-client llama-index-core llama-index-embeddings-huggingface llama-index-llms-groq python-dotenv
```

### Required Environment Variables (in `.env` file)
```
GROQ_API_KEY=your_groq_api_key
WEAVIATE_URL=your_weaviate_cloud_url
WEAVIATE_API_KEY=your_weaviate_api_key
```

### Sample Data
You need a `sample_data/` folder with text/document files (`.txt`, `.pdf`, `.docx`, etc.)

---

## Code Breakdown

### Step 1: Import Libraries

```python
from llama_index.core import Settings, VectorStoreIndex, StorageContext, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from dotenv import load_dotenv
import weaviate
from weaviate.classes.init import Auth
import os
```

**What each import does:**
- `llama_index.core`: Core RAG framework components
- `HuggingFaceEmbedding`: Converts text to vectors using pre-trained models
- `Groq`: LLM provider for generating responses
- `dotenv`: Loads environment variables from `.env` file
- `weaviate`: Vector database client
- `Auth`: Authentication handler for Weaviate

---

### Step 2: Load Environment Variables

```python
load_dotenv()
```

**Purpose:** Reads your `.env` file and makes `GROQ_API_KEY`, `WEAVIATE_URL`, and `WEAVIATE_API_KEY` available via `os.environ.get()`

**Why secure?** Never hardcode API keys in your code!

---

### Step 3: Configure LLM and Embeddings

```python
llm = Groq(model="llama-3.1-8b-instant", api_key=os.environ.get("GROQ_API_KEY"))
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2")
Settings.llm = llm
Settings.embed_model = embed_model
```

**Breakdown:**

| Component | Purpose |
|-----------|---------|
| `Groq(model=...)` | Initializes the LLM that will answer your questions |
| `HuggingFaceEmbedding(...)` | Initializes the embedding model that converts text → vectors |
| `Settings.llm` | Sets the LLM globally for all LlamaIndex operations |
| `Settings.embed_model` | Sets the embedding model globally |

**Models:**
- **LLM**: `llama-3.1-8b-instant` is fast and free via Groq
- **Embeddings**: `all-mpnet-base-v2` creates 768-dimensional vectors (good balance of speed/quality)

---

### Step 4: Connect to Weaviate Cloud

```python
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=os.environ.get("WEAVIATE_URL"),
    auth_credentials=Auth.api_key(os.environ.get("WEAVIATE_API_KEY"))
)
```

**Purpose:** Establishes connection to your Weaviate vector database in the cloud

**What is Weaviate?**
- A vector database optimized for similarity search
- Stores embeddings and retrieves them by semantic similarity
- Fast and scalable for production use

---

### Step 5: Load Documents from Disk

```python
documents = SimpleDirectoryReader("sample_data").load_data()
```

**Purpose:** Reads all documents from the `sample_data/` folder

**Supported formats:**
- `.txt` (plain text)
- `.pdf` (PDFs)
- `.docx` (Word documents)
- `.md` (Markdown)
- `.html` (HTML files)
- And many more!

**Output:** List of `Document` objects with:
- `doc.get_content()`: The actual text
- `doc.metadata`: File name, date, etc.

---

### Step 6: Prepare Collection (Database Table)

```python
collection_name = "RAGWeaviate"

# Delete if exists
try:
    client.collections.delete(collection_name)
except:
    pass
```

**Purpose:** 
- Deletes the old collection if it exists (fresh start)
- Wrapped in try-except to avoid errors if collection doesn't exist

**Why delete?** Ensures you're working with fresh data without duplicates

---

### Step 7: Create New Weaviate Collection

```python
from weaviate.classes.config import Configure, Property, DataType

client.collections.create(
    name=collection_name,
    vectorizer_config=Configure.Vectorizer.none(),
    properties=[
        Property(name="content", data_type=DataType.TEXT),
        Property(name="file_name", data_type=DataType.TEXT),
    ]
)
```

**Breakdown:**

| Parameter | Purpose |
|-----------|---------|
| `name=collection_name` | Names the collection "RAGWeaviate" |
| `vectorizer_config=Configure.Vectorizer.none()` | We'll provide embeddings ourselves (not using Weaviate's vectorizer) |
| `properties=[...]` | Defines what fields each document will have |

**Properties:**
- `content`: The actual document text (searchable)
- `file_name`: Where the document came from (metadata)

---

### Step 8: Generate Embeddings and Index Documents

```python
collection = client.collections.get(collection_name)

for doc in documents:
    # Get embedding
    embedding = embed_model.get_text_embedding(doc.get_content())
    
    collection.data.insert(
        properties={
            "content": doc.get_content(),
            "file_name": doc.metadata.get("file_name", "unknown")
        },
        vector=embedding
    )

print(f"✓ Indexed {len(documents)} documents")
```

**What happens here:**

1. **Get collection reference:** `client.collections.get(collection_name)`
2. **Loop through each document**
3. **Generate embedding:** Convert text to a vector (e.g., 768 numbers)
   - Example: `"What is AI?" → [0.234, -0.891, 0.445, ..., 0.123]`
4. **Insert into Weaviate:**
   - Store the embedding (for similarity search)
   - Store the properties (content and file_name)

**How embeddings work:**
- Similar texts have similar vectors
- "AI is important" and "Artificial intelligence matters" → very close vectors
- Used for semantic similarity search (not keyword matching)

---

### Step 9: Create Query Embedding

```python
query_text = "What are the different challenges in Copilots?"
query_embedding = embed_model.get_text_embedding(query_text)
```

**Purpose:** Convert the user's question into the same embedding space as the documents

**Key insight:** The question is embedded using the **same model** that embedded the documents, so they're comparable

---

### Step 10: Search for Similar Documents

```python
results = collection.query.near_vector(
    near_vector=query_embedding,
    limit=3,
    return_metadata=weaviate.classes.query.MetadataQuery(distance=True)
)
```

**Breakdown:**

| Parameter | Purpose |
|-----------|---------|
| `near_vector=query_embedding` | Find documents with similar embeddings |
| `limit=3` | Return top 3 most relevant results |
| `return_metadata=...` | Include distance scores (how similar) |

**How it works:**
- Weaviate calculates distance between query vector and all document vectors
- Returns the 3 documents with smallest distances (most similar)
- Distance score: lower = more similar (0 = identical, 1 = opposite)

---

### Step 11: Display Results

```python
print("\nSearch Results:")
for result in results.objects:
    print(f"- {result.properties['content'][:]}...")
```

**Purpose:** Show the retrieved documents to the user

**Output example:**
```
Search Results:
- Copilot challenges include latency, accuracy, and user adoption. Latency refers to...
- Microsoft Copilot faces integration difficulties with legacy systems. Many enterprises...
- AI copilots struggle with context understanding in complex domains...
```

---

## How It Works

### The Complete RAG Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG SYSTEM FLOW                           │
└─────────────────────────────────────────────────────────────┘

1. INDEXING PHASE (One-time setup)
   ├── Load Documents
   │   └── "sample_data/document.txt" → Content
   ├── Generate Embeddings
   │   └── Content → Vector (768 dimensions)
   └── Store in Weaviate
       └── Vector + Metadata → Database

2. QUERYING PHASE (Real-time)
   ├── User asks Question
   │   └── "What are Copilot challenges?"
   ├── Embed the Question
   │   └── Question → Vector (same 768 dimensions)
   ├── Vector Search in Weaviate
   │   └── Find 3 most similar documents
   ├── Retrieve Top Results
   │   └── Documents + Embeddings
   └── (Optional) Pass to LLM
       └── LLM generates response based on docs
```

### Example Walkthrough

**Your documents:**
```
Doc 1: "Copilots face latency issues when processing large datasets..."
Doc 2: "Weather in Mumbai is sunny today..."
Doc 3: "Copilot accuracy depends on training data quality..."
```

**Your question:**
```
"What are the different challenges in Copilots?"
```

**Embeddings (simplified):**
```
Doc 1 vector: [0.8, 0.7, 0.2, ...]  → Copilot-related
Doc 2 vector: [0.1, 0.2, 0.9, ...]  → Weather-related
Doc 3 vector: [0.9, 0.6, 0.1, ...]  → Copilot-related
Query vector: [0.7, 0.65, 0.15, ...] → About Copilots
```

**Search results (by distance):**
```
1st: Doc 3 (distance: 0.05) ✓ Most relevant
2nd: Doc 1 (distance: 0.12) ✓ Also relevant
3rd: Doc 2 (distance: 0.95) ✗ Not relevant
```

---

## Advanced Usage

### Option 1: Get LLM Answers Based on Retrieved Documents

```python
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.weaviate import WeaviateVectorStore

# Create index from Weaviate
vector_store = WeaviateVectorStore(
    weaviate_client=client,
    index_name="RAGWeaviate"
)

index = VectorStoreIndex.from_vector_store(vector_store)

# Query with LLM synthesis
query_engine = index.as_query_engine()
response = query_engine.query("What are the different challenges in Copilots?")
print(response)
```

**Difference:**
- Previous code: Returns raw documents
- This code: LLM reads documents and generates a synthesized answer

---

### Option 2: Batch Insert Faster

```python
# More efficient for large datasets
with collection.batch.dynamic() as batch:
    for doc in documents:
        embedding = embed_model.get_text_embedding(doc.get_content())
        batch.add_object(
            properties={
                "content": doc.get_content(),
                "file_name": doc.metadata.get("file_name", "unknown")
            },
            vector=embedding
        )
```

---

### Option 3: Add More Metadata

```python
client.collections.create(
    name="RAGWeaviate_Advanced",
    vectorizer_config=Configure.Vectorizer.none(),
    properties=[
        Property(name="content", data_type=DataType.TEXT),
        Property(name="file_name", data_type=DataType.TEXT),
        Property(name="created_date", data_type=DataType.TEXT),
        Property(name="author", data_type=DataType.TEXT),
    ]
)
```

---

## Common Issues & Solutions

### Issue 1: `ImportError: cannot import name '_ContextManagerWrapper'`

**Cause:** Broken `llama-index-vector-stores-weaviate` package

**Solution:**
```bash
pip uninstall llama-index-vector-stores-weaviate -y
pip install --upgrade weaviate-client llama-index-core
```

---

### Issue 2: "No documents found in sample_data"

**Cause:** Folder doesn't exist or is empty

**Solution:**
```bash
# Create folder
mkdir sample_data

# Add some .txt files with content
echo "Document content here" > sample_data/doc1.txt
```

---

### Issue 3: "Connection refused" to Weaviate

**Cause:** Wrong URL or API key

**Solution:**
```python
# Verify credentials
print(os.environ.get("WEAVIATE_URL"))
print(os.environ.get("WEAVIATE_API_KEY"))

# Test connection
try:
    client = weaviate.connect_to_weaviate_cloud(...)
    print("✓ Connected!")
except Exception as e:
    print(f"✗ Error: {e}")
```

---

### Issue 4: Slow embedding generation

**Cause:** HuggingFace model downloads for first time

**Solution:**
- First run takes time (model is ~400MB)
- Subsequent runs use cached model
- To speed up: Use smaller model like `all-MiniLM-L6-v2` instead

```python
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
```

---

### Issue 5: High costs with Groq API

**Cause:** Many API calls

**Solution:**
- Groq has generous free tier
- Cache embeddings (don't re-embed same docs)
- Use cheaper models if acceptable
- Batch queries together

---

## Performance Tips

| Tip | Benefit |
|-----|---------|
| Use batch insertion | 10-100x faster indexing |
| Increase `limit` gradually | Find optimal retrieval size |
| Use smaller embedding model | Faster but less accurate |
| Cache embeddings | Avoid re-computing |
| Use hybrid search | Combine keyword + semantic |

---

## Summary

This RAG system:

1. **Indexes** your documents by converting them to embeddings
2. **Stores** embeddings in Weaviate for fast retrieval
3. **Searches** by finding semantically similar documents
4. **Retrieves** top relevant results for any query
5. **(Optional)** Feeds results to LLM for intelligent answers

**Key Concepts:**
- **Embeddings**: Numerical vectors representing text meaning
- **Semantic Search**: Finding similar meaning, not keyword matches
- **Vector Database**: Fast similarity search at scale
- **RAG**: Uses external knowledge to improve LLM answers

---

## Next Steps

1. ✅ Set up environment variables
2. ✅ Install required packages
3. ✅ Create `sample_data/` with documents
4. ✅ Run the code
5. 🚀 Experiment with different queries
6. 🚀 Add more documents
7. 🚀 Integrate with your application

---

## Resources

- [Weaviate Documentation](https://weaviate.io/developers/weaviate)
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [HuggingFace Embeddings](https://huggingface.co/models?task=sentence-similarity)
- [Groq API Docs](https://console.groq.com/docs)
