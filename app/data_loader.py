# creeptalk/app/data_loader.py

import pandas as pd
import numpy as np
import faiss

def load_dataset():
    """
    사전 구축된 creepypasta 데이터셋과 FAISS index, embedding을 로드합니다.
    반환:
        - df: pd.DataFrame (본문 포함)
        - index: faiss.Index (유사도 검색용)
        - embeddings: np.ndarray (optional, 사용 안 해도 무방)
    """
    df = pd.read_csv("dataset/creepypastas_filtered.csv")
    index = faiss.read_index("dataset/faiss_index.index")
    embeddings = np.load("dataset/creepypasta_embeddings.npy").astype("float32")
    return df, index, embeddings
