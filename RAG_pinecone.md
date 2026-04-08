# Retrieval-Augmented Generation (RAG) with Pinecone and LlamaIndex
## Complete Step-by-Step Guide

---

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Architecture Comparison](#architecture-comparison)
4. [Code Breakdown](#code-breakdown)
5. [Detailed Explanations](#detailed-explanations)
6. [How It Works](#how-it-works)
7. [Advanced Features](#advanced-features)
8. [Common Issues & Solutions](#common-issues--solutions)
9. [Performance Optimization](#performance-optimization)

---

## Overview

This code implements a **Retrieval-Augmented Generation (RAG)** system using **Pinecone** as the vector database:

### What This Code Does:
1. **Loads** documents from local files
2. **Converts** them into embeddings (768-dimensional vectors)
3. **Stores** embeddings in Pinecone (cloud-based vector database)
4. **Retrieves** relevant documents based on semantic similarity
5. **Synthesizes** answers using Groq LLM based on retrieved documents

### Why Pinecone?
- **Fully Managed**: No infrastructure to manage
- **Scalable**: Handles millions of vectors efficiently
- **Fast**: Sub-millisecond query latency
- **Serverless**: Pay only for what you use
- **Built-in Features**: Namespaces, filtering, hybrid search

### How It Differs from Weaviate:
| Feature | Pinecone | Weaviate |
|---------|----------|----------|
| **Deployment** | Cloud-only (SaaS) | Self-hosted or Cloud |
| **Management** | Fully managed | You manage infrastructure |
| **Latency** | Ultra-low (<50ms) | Low (depends on setup) |
| **Scaling** | Auto-scaling | Manual scaling |
| **Cost** | Pay-per-use | Free tier available |
| **Use Case** | Production apps | Development & research |

---

## Prerequisites

### Required Libraries
```bash
pip install pinecone-client llama-index-core llama-index-embeddings-huggingface llama-index-llms-groq llama-index-vector-stores-pinecone python-dotenv
```

### Required Environment Variables (in `.env` file)
```
GROQ_API_KEY=your_groq_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
```

### Setup Instructions:

**1. Get Groq API Key:**
- Visit: https://console.groq.com
- Sign up for free
- Copy your API key

**2. Get Pinecone API Key:**
- Visit: https://www.pinecone.io
- Sign up (free tier available)
- Create a new project
- Copy your API key

**3. Create `.env` file in your project:**
```bash
cat > .env << EOF
GROQ_API_KEY=your_key_here
PINECONE_API_KEY=your_key_here
EOF
```

### Sample Data
You need a `sample_data/` folder with documents:
```bash
mkdir sample_data
echo "Document content here..." > sample_data/doc1.txt
```

---

## Architecture Comparison

### Weaviate Architecture (Self-managed)
```
┌─────────────────┐
│  Your Code      │
└────────┬────────┘
         │
         ├─── Embeddings ───┐
         │                  │
         └─── Vectors ──────┤
                            │
         ┌──────────────────┘
         │
    ┌────▼─────┐
    │ Weaviate  │  (You manage)
    │ Database  │
    └───────────┘
```

### Pinecone Architecture (Fully Managed)
```
┌─────────────────┐
│  Your Code      │
└────────┬────────┘
         │
         ├─── Embeddings ───┐
         │                  │
         └─── Vectors ──────┤
                            │
         ┌──────────────────┘
         │
    ┌────▼──────────────┐
    │   Pinecone        │  (Managed by Pinecone Inc.)
    │   (Cloud SaaS)    │
    │   - Auto-scaling  │
    │   - Monitoring    │
    │   - Backups       │
    └───────────────────┘
```

---

## Code Breakdown

### Step 1: Import Libraries

```python
from llama_index.core import(
    Settings, 
    VectorStoreIndex, 
    StorageContext,
    SimpleDirectoryReader
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os
import time
```

**What each import does:**

| Import | Purpose |
|--------|---------|
| `llama_index.core` | RAG framework with indexing & querying |
| `HuggingFaceEmbedding` | Pre-trained model to convert text → vectors |
| `Groq` | Fast LLM provider for answer generation |
| `PineconeVectorStore` | LlamaIndex integration with Pinecone |
| `Pinecone` | Pinecone client for index management |
| `ServerlessSpec` | Configuration for serverless Pinecone |
| `load_dotenv` | Load environment variables securely |

---

### Step 2: Load Environment Variables

```python
load_dotenv()
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
```

**Purpose:**
- Reads your `.env` file
- Makes API keys available securely
- Never hardcodes secrets in code

**Why this matters:**
- ✅ Secure: Keys not in source code
- ✅ Portable: Works across machines
- ✅ Professional: Industry best practice

---

### Step 3: Initialize Pinecone Client

```python
pc = Pinecone(api_key = PINECONE_API_KEY)
```

**Purpose:** Creates a Pinecone client to manage your vector database

**What you can do with `pc`:**
- `pc.list_indexes()` - See all your indexes
- `pc.create_index()` - Create new index
- `pc.delete_index()` - Delete index
- `pc.Index(name)` - Connect to existing index

**Analogy:** Like logging into a database management system

---

### Step 4: Configure LLM and Embeddings

```python
llm = Groq(model = "llama-3.1-8b-instant")
embed_model = HuggingFaceEmbedding(model_name = "sentence-transformers/all-mpnet-base-v2")
Settings.llm = llm
Settings.embed_model = embed_model
```

**Breakdown:**

| Component | What It Does |
|-----------|--------------|
| `Groq(model=...)` | Initializes fast LLM for answer generation |
| `HuggingFaceEmbedding(...)` | Loads embedding model that converts text to 768-D vectors |
| `Settings.llm` | Sets LLM globally for all LlamaIndex operations |
| `Settings.embed_model` | Sets embedding model globally |

**Model Details:**
- **LLM**: `llama-3.1-8b-instant` 
  - 8 billion parameters
  - Fastest inference on Groq
  - Great for real-time applications
  - Free tier: 10,000 tokens/day

- **Embeddings**: `sentence-transformers/all-mpnet-base-v2`
  - Produces 768-dimensional vectors
  - Excellent semantic understanding
  - ~400MB download (cached locally)

---

### Step 5: Define Index Name

```python
index_name = "llama3-groq-pinecone"
```

**Purpose:** Names your Pinecone index (vector database)

**Naming Convention:**
- Use lowercase
- Use hyphens instead of underscores
- Keep it descriptive: `[model]-[use-case]-[provider]`

**Example naming:**
```python
index_name = "company-docs-rag"           # For company documents
index_name = "ecommerce-product-search"   # For product search
index_name = "customer-support-qa"        # For customer support
```

---

### Step 6: Check Existing Indexes

```python
existing_indexes = [
    index_info["name"] for index_info in pc.list_indexes()
]
```

**What this does:**
1. `pc.list_indexes()` - Gets all your Pinecone indexes
2. List comprehension extracts just the names
3. Stores names in `existing_indexes` list

**Output example:**
```python
existing_indexes = ["llama3-groq-pinecone", "embeddings-test", "old-project"]
```

**Why useful:**
- Avoid creating duplicate indexes
- Check what indexes you have
- Prevent errors when connecting

---

### Step 7: Create Pinecone Index (if not exists)

```python
if index_name not in existing_indexes: 
    pc.create_index(
        name = index_name, 
        dimension = 768, 
        metric = "cosine", 
        spec = ServerlessSpec(
                cloud = "aws", 
                region = "us-east-1"
        )
    )
```

**Breakdown:**

| Parameter | Purpose | Example |
|-----------|---------|---------|
| `name` | Index name | `"llama3-groq-pinecone"` |
| `dimension` | Vector size | `768` (must match embedding model) |
| `metric` | Distance metric | `"cosine"` (best for embeddings) |
| `cloud` | Cloud provider | `"aws"`, `"gcp"`, `"azure"` |
| `region` | Server location | `"us-east-1"` (closest to you = faster) |

**Key Concepts:**

**Dimension (768):**
- Each word is converted to 768 numbers
- Example: `"AI is great" → [0.234, -0.891, 0.445, ..., 0.123]` (768 values)
- Must match your embedding model
- `all-mpnet-base-v2` always produces 768-dimensional vectors

**Metric (cosine):**
- How Pinecone measures similarity between vectors
- **Cosine**: Best for text embeddings (0° = identical, 180° = opposite)
- **Euclidean**: Distance-based (0 = identical, ∞ = opposite)
- **Dot Product**: Fast but requires normalized vectors

**ServerlessSpec:**
- **Serverless**: Auto-scaling, no infrastructure management
- **AWS**: Pinecone runs on Amazon's servers
- **Region**: `us-east-1` (N. Virginia) recommended for lowest latency

---

### Step 8: Connect to Pinecone Index

```python
index = pc.Index(index_name)
time.sleep(1)
```

**Purpose:**
1. `pc.Index()` - Gets reference to your index
2. `time.sleep(1)` - Waits 1 second for index to be ready

**Why sleep?** 
- New indexes take a moment to initialize
- Prevents "index not ready" errors
- Good practice after creating indexes

---

### Step 9: Check Index Statistics

```python
index.describe_index_stats()
```

**Purpose:** Gets information about your index

**Output example:**
```python
{
    "namespaces": {"": {"vector_count": 0}},
    "dimension": 768,
    "index_fullness": 0.0,
    "total_vector_count": 0
}
```

**What it tells you:**
- `vector_count`: How many vectors are stored
- `dimension`: Size of each vector (should be 768)
- `index_fullness`: What % of capacity is used (free tier: limited)

---

### Step 10: Load Documents

```python
documents = SimpleDirectoryReader("sample_data").load_data()
```

**Purpose:** Reads all documents from `sample_data/` folder

**Supported formats:**
- `.txt` (plain text)
- `.pdf` (PDFs) ⭐ Most common
- `.docx` (Word documents)
- `.md` (Markdown)
- `.html` (HTML files)
- `.json` (JSON files)
- And many more!

**Output:** List of `Document` objects:
```python
documents = [
    Document(text="Content of file 1", metadata={"file_name": "doc1.txt"}),
    Document(text="Content of file 2", metadata={"file_name": "doc2.txt"}),
    ...
]
```

---

### Step 11: Create PineconeVectorStore

```python
vector_store = PineconeVectorStore(pinecone_index = index)
```

**Purpose:** Wraps your Pinecone index for use with LlamaIndex

**What it does:**
- Handles all communication with Pinecone
- Manages embeddings (converting docs to vectors)
- Manages storage (saving vectors to index)
- Manages retrieval (finding similar vectors)

**Analogy:** Like a translator between LlamaIndex and Pinecone

---

### Step 12: Create Storage Context

```python
storage_context = StorageContext.from_defaults(vector_store = vector_store)
```

**Purpose:** Tells LlamaIndex where to store documents

**What's happening:**
- `StorageContext` = settings for document storage
- `vector_store = vector_store` = use Pinecone for storage
- Default behavior: embed documents automatically

---

### Step 13: Index Documents

```python
index = VectorStoreIndex.from_documents(
    documents, storage_context = storage_context
)
```

**This is the critical step! Here's what happens:**

```
┌─────────────────────────────────────────────────────┐
│            DOCUMENT INDEXING PROCESS                │
└─────────────────────────────────────────────────────┘

For each document:
  1. Split document into chunks
  2. For each chunk:
     a. Generate embedding (text → 768-D vector)
     b. Upload vector + chunk to Pinecone
     c. Store metadata (file name, position, etc.)

Result: Pinecone index now contains:
  - 768-dimensional vectors
  - Original text chunks
  - Metadata for retrieval
```

**Example:**

Original document:
```
"AI is transforming industries. Machine learning 
improves predictions. Deep learning powers neural networks."
```

After indexing:
```
Chunk 1: "AI is transforming industries."
  Vector: [0.234, -0.891, 0.445, ..., 0.123]
  
Chunk 2: "Machine learning improves predictions."
  Vector: [0.245, -0.880, 0.450, ..., 0.125]
  
Chunk 3: "Deep learning powers neural networks."
  Vector: [0.250, -0.870, 0.455, ..., 0.130]
```

All stored in Pinecone!

---

### Step 14: Create Query Engine

```python
query_engine = index.as_query_engine()
```

**Purpose:** Prepares your index for querying

**What it does:**
- Sets up LLM for answer generation
- Configures retrieval settings
- Prepares prompt templates

**Analogy:** Like opening your database for searches

---

### Step 15: Query and Get Response

```python
response = query_engine.query("What are the different challenges in Copilots?")
print(response)
```

**Here's the complete flow:**

```
┌──────────────────────────────────────────────────────────┐
│              QUERY EXECUTION FLOW                         │
└──────────────────────────────────────────────────────────┘

1. User asks: "What are challenges in Copilots?"

2. Embed query (using same model as documents)
   → [0.240, -0.885, 0.450, ..., 0.124]

3. Search Pinecone for similar vectors
   → Find top K (usually 3-5) most relevant chunks

4. Retrieve chunks from Pinecone
   Chunk 1: "Copilots face latency issues..."
   Chunk 2: "User adoption is challenging..."
   Chunk 3: "Accuracy depends on training data..."

5. Send to Groq LLM:
   "Based on these documents:
   [chunks above]
   Answer: What are the challenges in Copilots?"

6. LLM generates response:
   "Copilots face several challenges including:
   1. Latency - slow response times
   2. User adoption - difficulty gaining acceptance
   3. Accuracy - depends on data quality"

7. Return response to user
```

---

## How It Works

### Complete RAG Pipeline with Pinecone

```
┌──────────────────────────────────────────────────────────────┐
│                  PINECONE RAG SYSTEM                          │
└──────────────────────────────────────────────────────────────┘

INDEXING PHASE (Setup - run once)
├─ Load Documents
│  └─ Read files from sample_data/
├─ Generate Embeddings
│  └─ Convert text → 768-D vectors (HuggingFace)
├─ Upload to Pinecone
│  └─ Store vectors + metadata in cloud
└─ Index is now ready for queries

QUERYING PHASE (Real-time - run many times)
├─ User asks question
├─ Embed question (same model as docs)
├─ Search Pinecone (cosine similarity)
├─ Get top-K similar chunks
├─ Pass chunks to Groq LLM
├─ LLM synthesizes answer
└─ Return answer to user
```

### Example Walkthrough

**Your documents:**
```
doc1.txt: "Copilots struggle with latency when handling large datasets"
doc2.txt: "Weather in Mumbai is sunny"
doc3.txt: "Copilot accuracy depends on training data quality"
```

**Your question:**
```
"What are Copilot challenges?"
```

**Step 1: Embed documents (during indexing)**
```
Doc 1 vector: [0.8, 0.7, 0.2, ...] → Copilot + latency
Doc 2 vector: [0.1, 0.2, 0.9, ...] → Weather + Mumbai
Doc 3 vector: [0.9, 0.6, 0.1, ...] → Copilot + accuracy
```

**Step 2: Embed question (during query)**
```
Query vector: [0.75, 0.65, 0.15, ...] → About Copilot challenges
```

**Step 3: Find similar vectors in Pinecone**
```
Similarity to Doc 1: 0.95 (very similar) ✓
Similarity to Doc 2: 0.05 (very different) ✗
Similarity to Doc 3: 0.92 (very similar) ✓
```

**Step 4: Retrieve top-2 documents**
```
1. Doc 3: "Copilot accuracy depends on training data quality"
2. Doc 1: "Copilots struggle with latency when handling datasets"
```

**Step 5: LLM synthesizes answer**
```
Input to Groq:
"Based on these documents:
1. Copilot accuracy depends on training data quality
2. Copilots struggle with latency when handling datasets
Answer: What are Copilot challenges?"

LLM Response:
"Copilot challenges include:
1. Accuracy - depends on training data quality
2. Latency - struggles with large datasets"
```

---

## Advanced Features

### Feature 1: Using Namespaces (Multi-tenant)

```python
# Index documents into different namespaces
vector_store = PineconeVectorStore(
    pinecone_index=index,
    namespace="customer_docs"  # Separate namespace
)

# Different namespace = different set of docs
# Useful for: multi-tenant apps, multiple projects, A/B testing
```

---

### Feature 2: Adjust Retrieval Settings

```python
# Get more or fewer results
query_engine = index.as_query_engine(
    similarity_top_k=5,  # Return top 5 results (default: 2)
    response_mode="compact"  # Compact or verbose responses
)
```

---

### Feature 3: Metadata Filtering

```python
# Filter documents by metadata during retrieval
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters

filters = MetadataFilters(
    filters=[ExactMatchFilter(key="category", value="copilots")]
)

response = query_engine.query(
    "What are challenges?",
    filters=filters  # Only search in 'copilots' category
)
```

---

### Feature 4: Add More Metadata to Documents

```python
# When loading documents, add custom metadata
from llama_index.core import Document

custom_docs = [
    Document(
        text="Copilot content...",
        metadata={
            "file_name": "copilots.txt",
            "category": "copilots",
            "date": "2024-04-08",
            "author": "Sachin"
        }
    )
]

# Then index with metadata
index = VectorStoreIndex.from_documents(
    custom_docs, storage_context=storage_context
)
```

---

### Feature 5: Batch Indexing for Large Datasets

```python
# More efficient for 1000+ documents
from llama_index.core import Document

# Process in batches
batch_size = 100
for i in range(0, len(documents), batch_size):
    batch = documents[i:i+batch_size]
    idx = VectorStoreIndex.from_documents(batch, storage_context=storage_context)
    print(f"Indexed batch {i//batch_size + 1}")
```

---

## Common Issues & Solutions

### Issue 1: `ModuleNotFoundError: No module named 'pinecone'`

**Cause:** Pinecone not installed

**Solution:**
```bash
pip install pinecone-client llama-index-vector-stores-pinecone
```

---

### Issue 2: `AuthenticationError: Invalid API key`

**Cause:** Wrong or missing API key

**Solution:**
```python
# Verify .env file exists and has correct key
import os
from dotenv import load_dotenv

load_dotenv()
print(os.environ.get("PINECONE_API_KEY"))  # Should print your key

# If empty, check:
# 1. .env file exists in project root
# 2. Format: PINECONE_API_KEY=your_actual_key
# 3. No quotes around the key
```

---

### Issue 3: `ValueError: Dimension mismatch`

**Cause:** Embedding model dimension doesn't match index dimension

**Solution:**
```python
# Your index has dimension 768
# But you're using embedding model with different dimension

# Fix: Use correct model
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# ✓ Correct (produces 768-D vectors)
embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# ✓ Alternative (produces 384-D vectors)
embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
# Then create index with dimension=384
```

---

### Issue 4: `TimeoutError: Index not ready`

**Cause:** Index creation takes time

**Solution:**
```python
import time

# Wait longer after creating index
if index_name not in existing_indexes:
    pc.create_index(...)
    time.sleep(5)  # Increase from 1 to 5 seconds

# Or check status
while not pc.Index(index_name).describe_index_stats():
    time.sleep(1)
    print("Waiting for index...")
```

---

### Issue 5: `No documents found in sample_data`

**Cause:** Folder doesn't exist or is empty

**Solution:**
```bash
# Create sample_data folder
mkdir sample_data

# Add test file
echo "Sample document about AI and machine learning" > sample_data/test.txt

# Verify
ls sample_data/
```

---

### Issue 6: Slow embedding generation on first run

**Cause:** HuggingFace downloads model (~400MB)

**Solution:**
```python
# First run takes 2-5 minutes (downloads model)
# Subsequent runs use cached model (< 1 second)

# To pre-download model:
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-mpnet-base-v2"
)
# This caches the model

# Or use a smaller model (faster):
embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"  # 384-D, faster
)
```

---

## Performance Optimization

### Speed Optimization

| Optimization | Impact | How |
|--------------|--------|-----|
| Use smaller embedding model | 5-10x faster | Use `all-MiniLM-L6-v2` instead of `all-mpnet-base-v2` |
| Batch indexing | 2-3x faster | Index 100 docs at once instead of 1 |
| Use caching | 10-100x faster | Cache embeddings, reuse vectors |
| Increase `similarity_top_k` | Faster retrieval | Get more results per query (default: 2) |
| Use namespaces | Faster searches | Split docs into separate namespaces |

### Cost Optimization

| Strategy | Benefit | Implementation |
|----------|---------|-----------------|
| Delete unused indexes | Reduce storage costs | `pc.delete_index(name)` |
| Use free tier wisely | $0 cost | Limited to 1 index, 2GB storage |
| Reuse index for multiple projects | Share storage | Use namespaces |
| Delete old vectors | Reduce size | Don't keep duplicates |
| Compress documents | Smaller vectors | Summarize documents first |

### Quality Optimization

| Strategy | Benefit | How |
|----------|---------|-----|
| Better chunking | More relevant retrieval | Split documents semantically |
| More metadata | Better filtering | Add category, date, author, etc. |
| Larger `similarity_top_k` | More context for LLM | Increase from 2 to 5 |
| Better prompt engineering | Better answers | Improve LLM system prompt |
| Reranking | Filter irrelevant docs | Use semantic reranking before LLM |

---

## Pinecone vs Weaviate Comparison

| Feature | Pinecone | Weaviate |
|---------|----------|----------|
| **Deployment** | Cloud-only (SaaS) | Self-hosted or Cloud |
| **Setup time** | 5 minutes | 30+ minutes |
| **Infrastructure** | Zero (managed) | You manage |
| **Latency** | Ultra-low (<50ms) | Low (variable) |
| **Scaling** | Auto | Manual |
| **Free tier** | Yes (1 index, 2GB) | Yes (free tier) |
| **Cost** | Pay-per-use | Free to expensive |
| **Best for** | Production apps | Development |
| **Namespaces** | Yes (multi-tenant) | No |
| **Metadata filtering** | Yes | Yes |
| **Hybrid search** | Yes (Pro plan) | Yes |
| **Learning curve** | Easy | Medium |

---

## Pinecone Best Practices

### ✅ DO:

```python
# 1. Use environment variables for secrets
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]

# 2. Check if index exists before creating
if index_name not in existing_indexes:
    pc.create_index(...)

# 3. Add meaningful metadata
metadata = {"file_name": "doc.txt", "category": "ai", "date": "2024"}

# 4. Use namespaces for multi-tenant apps
vector_store = PineconeVectorStore(
    pinecone_index=index,
    namespace="customer_123"
)

# 5. Monitor index statistics
stats = index.describe_index_stats()
print(f"Vector count: {stats['total_vector_count']}")
```

### ❌ DON'T:

```python
# 1. Hardcode API keys
PINECONE_API_KEY = "pc_abc123xyz"  # Bad!

# 2. Create duplicate indexes
if True:
    pc.create_index(name)  # Creates every time!

# 3. Forget to add metadata
metadata = {}  # No info about document

# 4. Mix namespaces
# Don't index some docs to "users" and others to "products" if unintended

# 5. Ignore costs
# Monitor free tier usage - 2GB limit!
```

---

## Complete End-to-End Workflow

```python
# 1. Setup (run once)
load_dotenv()
pc = Pinecone(api_key=PINECONE_API_KEY)

# 2. Create index (run once)
if index_name not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(name=index_name, dimension=768, metric="cosine", 
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"))
    time.sleep(1)

# 3. Index documents (run when data changes)
documents = SimpleDirectoryReader("sample_data").load_data()
vector_store = PineconeVectorStore(pinecone_index=pc.Index(index_name))
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

# 4. Query (run for each user question)
query_engine = index.as_query_engine()
response = query_engine.query("What are Copilot challenges?")
print(response)

# 5. Cleanup (when done)
# pc.delete_index(index_name)  # Uncomment to delete
```

---

## Resources

- [Pinecone Documentation](https://docs.pinecone.io)
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [HuggingFace Embeddings](https://huggingface.co/models?task=sentence-similarity)
- [Groq API Documentation](https://console.groq.com/docs)
- [RAG Best Practices](https://docs.llamaindex.ai/en/stable/concepts/rag/)

---

## Summary

This code implements a production-ready RAG system with:

1. **Indexing**: Documents → Embeddings → Pinecone
2. **Retrieval**: Query → Embedding → Semantic Search → Top K docs
3. **Generation**: Retrieved docs → LLM → Answer
4. **Optimization**: Fast, scalable, cost-effective

**Key Takeaways:**
- Pinecone is fully managed (no infrastructure)
- Embeddings enable semantic (meaning-based) search
- RAG allows LLMs to use your custom data
- Groq provides fast, free LLM inference

**Next Steps:**
1. ✅ Get API keys (Groq + Pinecone)
2. ✅ Install libraries
3. ✅ Create `.env` file
4. ✅ Prepare sample documents
5. ✅ Run the code
6. 🚀 Experiment with different queries
7. 🚀 Customize for your use case
