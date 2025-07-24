import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

# # 데이터 불러오기
# df = pd.read_csv("dataset/creepypastas_filtered.csv")
# index = faiss.read_index("dataset/faiss_index.index")
# embed = np.load("dataset/creepypasta_embeddings.npy").astype("float32")

# # 임베딩 모델
# embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# # LLM 로딩
# model_id = "meta-llama/Llama-3.2-1B-Instruct"
# tokenizer = AutoTokenizer.from_pretrained(model_id, token="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")
# model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto", token="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")
# llm = pipeline("text-generation", model=model, tokenizer=tokenizer)

# def query_rag_system(query, top_k=3):
#     query_vec = embedder.encode([query]).astype("float32")
#     distances, indices = index.search(query_vec, top_k)
#     context = "\n\n".join(df.iloc[i]["body"] for i in indices[0])

#     prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
# You are a helpful assistant that answers questions based on scary stories.

# <|start_header_id|>user<|end_header_id|>
# [CONTEXT]
# {context}

# [QUESTION]
# {query}

# <|start_header_id|>assistant<|end_header_id|>"""

#     response = llm(prompt, max_new_tokens=300, temperature=0.7)[0]["generated_text"]
#     return response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
