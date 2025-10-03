# embed_tester.py

import time
import requests
import numpy as np
from typing import List, Tuple

# ---- Configure once here ----
BASE_URL = "http://127.0.0.1:9005"
DEFAULT_MODELS = ["nomic", "e5-large-v2", "mxbai", "arctic", "bge-m3"]

# Sample texts (replace with your own docs/query if you like)
DOCS = [
    "Password resets are handled in Okta under Settings > Account.",
    "To change MFA, open Security > Multifactor in your Okta dashboard.",
    "For SSO issues, contact the IT helpdesk or check the status page."
]
QUERY = "How do I reset my Okta password?"

# ---- Helpers ----
def post_embed(model: str, texts: List[str], mode: str = "document",
               batch_size: int = 64, normalize: bool = True, timeout: int = 120
               ) -> Tuple[np.ndarray, int, float]:
    t0 = time.perf_counter()
    r = requests.post(
        f"{BASE_URL}/embed",
        params={"model": model},
        json={"texts": texts, "mode": mode, "batch_size": batch_size, "normalize": normalize},
        timeout=timeout,
    )
    r.raise_for_status()
    data = r.json()
    vecs = np.array(data["vectors"], dtype=np.float32)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    return vecs, int(data["dims"]), dt_ms

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def vector_norm_ok(v: np.ndarray, tol: float = 1e-2) -> Tuple[bool, float]:
    n = float(np.linalg.norm(v))
    return (abs(n - 1.0) < tol, n)

# ---- Public API ----
def test_one_model(model: str, docs: List[str] = DOCS, query: str = QUERY) -> None:
    """Embed documents + query for a single model and print ranked results."""
    # docs
    doc_vecs, dims_docs, t_docs = post_embed(model, docs, mode="document")
    # query
    q_vec, dims_query, t_query = post_embed(model, [query], mode="query")

    if dims_docs != dims_query:
        print(f"[{model}] WARN dims mismatch: docs={dims_docs}, query={dims_query}")

    scores = [cosine(q_vec[0], d) for d in doc_vecs]
    order = np.argsort(scores)[::-1]
    ok, n = vector_norm_ok(q_vec[0])

    print(f"\n=== {model} ===")
    print(f"dims={dims_docs}  docs_time={t_docs:.1f}ms  query_time={t_query:.1f}ms  |  norm={n:.4f} ({'OK' if ok else 'not normalized'})")
    for rank, idx in enumerate(order, start=1):
        preview = docs[idx][:80].replace("\n", " ")
        print(f"  {rank}. {scores[idx]:.4f}  ::  {preview}")

def test_all(models: List[str] = None, docs: List[str] = DOCS, query: str = QUERY) -> None:
    """Loop through models and run test_one_model() for each."""
    models = models or DEFAULT_MODELS

    # quick health check
    try:
        health = requests.get(f"{BASE_URL}/healthz", timeout=10).json()
        print(f"Health: {health}")
    except Exception as e:
        print(f"Server not reachable at {BASE_URL}: {e}")
        return

    for m in models:
        try:
            test_one_model(m, docs=docs, query=query)
        except requests.HTTPError as e:
            detail = getattr(getattr(e, "response", None), "text", str(e))
            print(f"[{m}] HTTP error: {detail[:300]}")
        except Exception as e:
            print(f"[{m}] Error: {e}")
