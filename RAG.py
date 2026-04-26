from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

docs = ["AI is transforming India", "LLMs are powerful", "Deep learning is everywhere"]

# Embed
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(docs)

# Store in FAISS
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

# Query
query = "What is AI doing in India?"
q_embed = model.encode([query])

_, I = index.search(np.array(q_embed), k=2)
results = [docs[i] for i in I[0]]

print("Relevant:", results)