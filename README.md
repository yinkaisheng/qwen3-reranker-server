# Qwen3-Reranker Server

A reranking service that loads the [Qwen3-Reranker](https://modelscope.cn/models/Qwen/Qwen3-Reranker-0.6B) model with vLLM and wraps it with FastAPI. The API follows the Cohere/Jina-style Rerank API standard and can be seamlessly integrated into mainstream RAG frameworks such as Dify, LangChain, and LlamaIndex.

## Quick Start

```bash
pip install -r requirements.txt
CUDA_VISIBLE_DEVICES=0 python qwen3_reranker_server.py
```

Options:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `-p`, `--port` | Server port | 9994 |
| `-mp`, `--model_path` | Model path | `~/.cache/modelscope/.../Qwen3-Reranker-0.6B` |
| `-mml`, `--max_model_length` | Max context length | 8192 |
| `-gmu`, `--gpu_memory_utilization` | GPU memory utilization | 0.2 |
| `-ss`, `--swap_space` | CPU swap space (GB) | 4 |
| `-eg`, `--eager_mode` | Eager mode (faster startup, slower inference) | False |

## API

```bash
curl -X POST http://localhost:9994/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-Reranker-0.6B",
    "query": "What is the capital of China?",
    "documents": [
      "Beijing is the capital of China.",
      "Nanjing is a city of China."
    ]
  }'
```

Response:

```json
{
  "id": "rerank-<uuid>",
  "results": [
    {"index": 0, "relevance_score": 0.9987, "token_count": 42},
    {"index": 1, "relevance_score": 0.0012, "token_count": 38}
  ]
}
```
