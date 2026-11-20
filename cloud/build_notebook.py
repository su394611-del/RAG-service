"""Generate notebooks/RAG_Service_Colab_Kaggle.ipynb from cloud/rag_cloud.py.

The notebook embeds rag_cloud.py with %%writefile, so it runs even where the
repository cannot be downloaded (for example Kaggle without internet).

    python cloud/build_notebook.py            # write the notebook
    python cloud/build_notebook.py --check    # exit 1 if the notebook is out of date
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "cloud" / "rag_cloud.py"
OUT = ROOT / "notebooks" / "RAG_Service_Colab_Kaggle.ipynb"
GH = "su394611-del/RAG-service"
NB_PATH = "notebooks/RAG_Service_Colab_Kaggle.ipynb"
COLAB = f"https://colab.research.google.com/github/{GH}/blob/main/{NB_PATH}"
KAGGLE = f"https://www.kaggle.com/kernels/welcome?src=https://github.com/{GH}/blob/main/{NB_PATH}"
BINDER = f"https://mybinder.org/v2/gh/{GH}/HEAD?labpath={NB_PATH.replace('/', '%2F')}"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def build() -> dict:
    source = SRC.read_text(encoding="utf-8")
    cells = [
        md(f"""
# Multi-Tenant RAG Service — run it on Colab, Kaggle or Binder

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB})
[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)]({KAGGLE})
[![Launch Binder](https://mybinder.org/badge_logo.svg)]({BINDER})

This notebook loads **public Wikipedia passages** ([`{'rag-datasets/rag-mini-wikipedia'}`](https://huggingface.co/datasets/rag-datasets/rag-mini-wikipedia), CC BY 3.0) plus two **fictional tenant knowledge bases**, embeds them with **`BAAI/bge-small-en-v1.5`**, stores them in a **Qdrant** vector database (in memory), and answers questions with a hard `tenant_id` filter, a semantic cache, PII redaction and a prompt-injection check. It then measures retrieval accuracy on the dataset's question/answer pairs.

- **No API keys and no Hugging Face, Qdrant or OpenAI account.** The dataset and model download anonymously.
- Colab and Kaggle need their own sign-in to *run* code; **Binder needs none** (it can take a few minutes to start).
- **Run all cells** (Colab: *Runtime → Run all*). CPU is enough: about 2–4 minutes for the full corpus on CPU, under a minute on a GPU.
- **Kaggle:** switch on *Internet* in *Notebook options* for the full run. Without internet the notebook still runs, using a bundled corpus, hashing embeddings and a NumPy vector store, and says so.
- Optional last cell: a generated, cited answer. In Colab it uses Google's **free Gemini** models through `google.colab.ai` with no key; with your own `ANTHROPIC_API_KEY` it uses **Claude**; `GEMINI_API_KEY` works anywhere. Keys are never saved in the notebook.

The saved outputs below come from an automated run on GitHub Actions (CPU). Source: [`cloud/rag_cloud.py`](https://github.com/{GH}/blob/main/cloud/rag_cloud.py).
"""),
        md("## 1. Write the service module\nThe full source of `cloud/rag_cloud.py`, embedded so the notebook needs nothing else from the repository."),
        code("%%writefile rag_cloud.py\n" + source),
        md("## 2. Environment and packages\nInstalls only what is missing (`qdrant-client`, `sentence-transformers`, `pyarrow`). Nothing here needs a login."),
        code("""
import json, sys
from collections import Counter
import rag_cloud as rc

ENV, ONLINE = rc.detect_environment(), rc.internet_ok()
print(f"Environment: {ENV} | internet: {'yes' if ONLINE else 'no'} | Python {sys.version.split()[0]}")
if ONLINE:
    print(rc.ensure_packages({"qdrant_client": "qdrant-client>=1.12",
                              "sentence_transformers": "sentence-transformers>=3.4",
                              "pyarrow": "pyarrow"}))
else:
    print("No internet: using the bundled corpus, hashing embeddings and a NumPy vector store.")
"""),
        md("## 3. Load the data\nTwo fictional tenants (`tenant_a`, `tenant_b`) plus the public corpus under tenant `wiki`. Long passages are split into overlapping chunks."),
        code("""
DOCS, QA, DATA_INFO = rc.load_documents("auto")
print(json.dumps(DATA_INFO, indent=1))
print("Documents per tenant:", dict(Counter(d.tenant_id for d in DOCS)))
print("Example passage:", DOCS[len(DOCS) // 2].text[:300])
"""),
        md("## 4. Embed and index in the vector database\nBGE embeddings (normalised, cosine) upserted into a Qdrant collection with a `tenant_id` keyword payload."),
        code("""
service, INDEX_INFO = rc.RAGCloudService.build(DOCS, progress=print)
print(json.dumps(INDEX_INFO, indent=1))
"""),
        md("## 5. Demo: tenant isolation, cache, guards\nEvery query is filtered by tenant. `cross_tenant` counts passages from another tenant and must be 0."),
        code("""
import pandas as pd
pd.set_option("display.max_colwidth", 90)
rows = rc.run_demo(service)
display(pd.DataFrame(rows)[["purpose", "tenant", "answer", "sources", "cached", "blocked", "pii", "cross_tenant"]])
print("Cross-tenant passages returned:", sum(r["cross_tenant"] for r in rows))
"""),
        md("## 6. Retrieval evaluation\nFor 200 dataset questions (yes/no answers excluded): is the reference answer inside a retrieved passage? Reports hit@1, hit@5 and MRR."),
        code("""
RESULT = rc.evaluate(service, QA, n=200, top_k=5)
print(json.dumps(RESULT, indent=1))
"""),
        md("## 7. Ask your own question\nEdit `QUESTION` and `TENANT` (`wiki`, `tenant_a` or `tenant_b`) and run the cell."),
        code("""
QUESTION = "Who invented the voltaic pile?"
TENANT = "wiki"
r = service.handle(rc.RAGRequest(QUESTION, TENANT, top_k=3))
print("Answer :", r.answer)
print("Sources:", ", ".join(r.sources) or "-")
for p in r.passages:
    print(f"  {p['score']:.3f}  [{p['doc_id']}]  {p['text'][:160]}")
"""),
        md(f"""
## 8. Optional: generated, cited answer
The service itself is extractive. This cell adds a grounded answer that cites passage IDs, using the first available of:
1. **Claude** (`{'claude-sonnet-5-5'}`) when `ANTHROPIC_API_KEY` is set — Colab or Kaggle *Secrets*, or the prompt below;
2. **Google's free Gemini models** through `google.colab.ai` — automatic in Colab, no key needed;
3. the **Gemini API** when `GEMINI_API_KEY` is set (free key from Google AI Studio).

Keys stay in this session's memory and are not written to the notebook. Set `ASK_FOR_KEY = True` to type a Claude key.
"""),
        code("""
ASK_FOR_KEY = False
if ASK_FOR_KEY and not rc.claude_key():
    import getpass, os
    os.environ["ANTHROPIC_API_KEY"] = getpass.getpass("ANTHROPIC_API_KEY (not saved): ")

options = rc.generator_options()
print("Available generators:", options or "none")
if options:
    r = service.handle(rc.RAGRequest(QUESTION, TENANT, top_k=3), generate=True)
    print(f"\\n{r.generator}:\\n{r.generated}")
else:
    print("No generator here, so the answer above stays extractive. Open this notebook in Colab for free Gemini,"
          " or set ANTHROPIC_API_KEY / GEMINI_API_KEY.")
"""),
        md("""
## Notes and limits
- Tenant documents are **fictional**. Wikipedia passages come from `rag-mini-wikipedia` (CC BY 3.0).
- PII and prompt-injection checks are illustrative regex heuristics, not complete defences.
- Qdrant runs in local mode inside this process; pass `path=` to `rc.make_store("qdrant", path)` to persist it on disk.
- For the HTTP API with per-tenant bearer keys, run `RAG_Service_Corrected_2026_Diagnostics.py --api` from the repository.
"""),
    ]
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}, "colab": {"provenance": []},
                         "kaggle": {"accelerator": "none", "isInternetEnabled": True}}}


def sources_of(nb: dict) -> list[str]:
    return ["".join(c["source"]) for c in nb["cells"]]


def main() -> int:
    nb = build()
    if "--check" in sys.argv:
        if not OUT.is_file():
            print("notebook missing")
            return 1
        same = sources_of(json.loads(OUT.read_text(encoding="utf-8"))) == sources_of(nb)
        print("notebook up to date" if same else "notebook is out of date: run python cloud/build_notebook.py")
        return 0 if same else 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(nb['cells'])} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
