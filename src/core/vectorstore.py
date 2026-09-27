import hashlib
import numpy as np
import time
from pathlib import Path
from pymilvus import MilvusClient
from src.config import MILVUS_URI, MILVUS_TOKEN, MILVUS_DB, EMBED_MODEL, EMBED_URL, DIM
from src.core.llm import post_with_retry
from src.core.ledger import ledger

EMB = {}

def load_cache():
    path = Path("emb_cache.npz")
    if path.exists():
        z = np.load(path)
        EMB.update({str(k): v.astype(np.float32) for k, v in zip(z['keys'], z['vectors'])})
    return len(EMB)

def save_cache():
    np.savez_compressed("emb_cache.npz", keys=np.array(list(EMB)), vectors=np.stack(list(EMB.values())).astype(np.float16))

def embed_batch(texts):
    started = time.perf_counter()
    data = post_with_retry({"model": EMBED_MODEL, "input": texts, "dimensions": DIM}, EMBED_URL)
    ledger.add("embed", EMBED_MODEL, data.get("usage") or {}, time.perf_counter() - started)
    return [row["embedding"] for row in data["data"]]

def key_of(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def embed_cached(texts):
    missing = list(dict.fromkeys(t for t in texts if key_of(t) not in EMB))
    for i in range(0, len(missing), 64):
        batch = missing[i:i + 64]
        for text, vector in zip(batch, embed_batch(batch)):
            EMB[key_of(text)] = np.asarray(vector, dtype=np.float32)

    vectors = np.stack([EMB[key_of(t)] for t in texts])
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)

if MILVUS_DB:
    if MILVUS_DB not in client.list_databases():
        client.create_database(MILVUS_DB)
    client.use_database(MILVUS_DB)


def index_to_milvus(name, chunks, vectors):
    if client.has_collection(name):
        client.drop_collection(name)

    client.create_collection(name, dimension=vectors.shape[1], metric_type="COSINE")

    rows = [{"id": i, "vector": v.tolist(), **c} for i, (c, v) in enumerate(zip(chunks, vectors))]
    for i in range(0, len(rows), 1000):
        client.insert(name, rows[i:i + 1000])

    client.flush(name)
    return client.get_collection_stats(name)


def search_milvus(name, query, k=5, flt=""):
    vector = embed_cached([query])[0].tolist()
    hits = client.search(
        name,
        data=[vector],
        limit=k,
        filter=flt,
        output_fields=["text", "metadata"]
    )[0]

    return [
        {
            "id": h["id"],
            "score": round(h["distance"], 4),
            "text": h["entity"]["text"],
            **h["entity"]["metadata"]
        } for h in hits
    ]