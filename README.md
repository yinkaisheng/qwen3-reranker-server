# Qwen3-Reranker Server

> **This project is no longer maintained.** Since [v0.9.2](https://github.com/vllm-project/vllm/releases/tag/v0.9.2), vLLM natively supports [Qwen3-Reranker](https://modelscope.cn/models/Qwen/Qwen3-Reranker-0.6B) via [PR #19260](https://github.com/vllm-project/vllm/pull/19260) (not supported in v0.9.1 and earlier; see [issue #20532](https://github.com/vllm-project/vllm/issues/20532)). You can start a Rerank service directly with `vllm serve` — no FastAPI wrapper from this repo is needed.

This repository is kept for historical reference. For new deployments, use the native vLLM approach.

---

## Running Qwen3 Reranker with Native vLLM

### Command-Line Startup

```bash
export VLLM_USE_MODELSCOPE=True   # Enable when pulling models from ModelScope

vllm serve Qwen/Qwen3-Reranker-0.6B \
  --served-model-name Qwen3-Reranker-0.6B \
  --host 0.0.0.0 \
  --port 9995 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.06 \
  --chat-template /path/to/Qwen3-Reranker-0.6B/qwen3_reranker.jinja \
  --hf_overrides '{"architectures": ["Qwen3ForSequenceClassification"],"classifier_from_token": ["no", "yes"],"is_original_qwen3_reranker": true}'
```

Key parameters:

| Parameter | Description |
|-----------|-------------|
| `--hf_overrides` | **Required.** Routes the official Qwen3 Reranker to `Qwen3ForSequenceClassification` and extracts classification logits from `no`/`yes` tokens |
| `--chat-template` | Path to `qwen3_reranker.jinja` in the model directory for correct query/document formatting (see below) |
| `--max-model-len` | Maximum context length; default 8192 |
| `--gpu-memory-utilization` | GPU memory utilization ratio; the 0.6B model can use a low value (e.g. 0.06) |
| `VLLM_USE_MODELSCOPE=True` | Download models from ModelScope mirrors |

The `qwen3_reranker.jinja` file referenced by `--chat-template` (usually bundled with the model):

```jinja
<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: {{ instruction | default(instruct | default(messages | selectattr("role", "eq", "system") | map(attribute="content") | first | default("Given a web search query, retrieve relevant passages that answer the query", true), true), true) }}
<Query>: {{ messages | selectattr("role", "eq", "query") | map(attribute="content") | first }}
<Document>: {{ messages | selectattr("role", "eq", "document") | map(attribute="content") | first }}<|im_end|>
<|im_start|>assistant
<think>

</think>
```

> **Version requirement:** v0.9.2 minimum ([Release Notes](https://github.com/vllm-project/vllm/releases/tag/v0.9.2) · [Official Example](https://docs.vllm.ai/en/v0.9.2/examples/offline_inference/qwen3_reranker.html)). The Docker reference below uses v0.24.0.

### Docker Compose Reference

```yaml
services:
  qwen3-reranker-0.6b:
    image: vllm/vllm-openai:v0.24.0-aarch64-ubuntu2404
    container_name: qwen3-reranker-0.6b
    network_mode: host
    volumes:
      - ./Qwen:/Qwen
      - ./vllm_cache_0.24:/root/.cache/vllm
    #  - ./uvicorn_config_vllm_0.24.0.py:/usr/local/lib/python3.12/dist-packages/uvicorn/config.py
    #  - ./serving_vllm_0.24.0.py:/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/chat_completion/serving.py
    environment:
      - VLLM_USE_MODELSCOPE=True
      - TZ=Asia/Shanghai
    entrypoint: >
      vllm serve /Qwen/Qwen3-Reranker-0.6B
      --served-model-name Qwen3-Reranker-0.6B
      --host 0.0.0.0
      --port 9995
      --tensor-parallel-size 1
      --max-model-len 8192
      --gpu-memory-utilization 0.06
      --allowed-local-media-path /Qwen
      --enable-log-requests
      --chat-template /Qwen/Qwen3-Reranker-0.6B/qwen3_reranker.jinja
      --hf_overrides '{"architectures": ["Qwen3ForSequenceClassification"],"classifier_from_token": ["no", "yes"],"is_original_qwen3_reranker": true}'
    shm_size: '32gb'
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:9995/health || exit 1"]
      interval: 600s
      timeout: 10s
      retries: 3
      start_period: 600s
    deploy:
      resources:
        limits:
          memory: 64G
        reservations:
          devices:
            - capabilities: [gpu]
              driver: nvidia
              device_ids: ["0"]
    restart: unless-stopped
```

### API Usage

The native vLLM service exposes a Cohere/Jina-style Rerank API compatible with Dify, LangChain, LlamaIndex, and other RAG frameworks:

```bash
curl -X POST http://localhost:9995/v1/rerank \
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

Example response:

```json
{
  "id": "score-8c778b770824f730",
  "model": "Qwen3-Reranker-0.6B",
  "usage": {
    "prompt_tokens": 171,
    "total_tokens": 171
  },
  "results": [
    {
      "index": 0,
      "document": {
        "text": "Beijing is the capital of China.",
        "multi_modal": null
      },
      "relevance_score": 0.9956526756286621
    },
    {
      "index": 1,
      "document": {
        "text": "Nanjing is a city of China.",
        "multi_modal": null
      },
      "relevance_score": 0.039862848818302155
    }
  ]
}
```

---

## Historical Note (This Repository)

This repo previously wrapped Qwen3-Reranker with vLLM + FastAPI, providing a `/v1/rerank` endpoint before vLLM had native reranker support. See `qwen3_reranker_server.py` for the implementation — it is no longer needed.

<details>
<summary>Legacy startup (deprecated)</summary>

```bash
pip install -r requirements.txt
CUDA_VISIBLE_DEVICES=0 python qwen3_reranker_server.py
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `-p`, `--port` | Server port | 9994 |
| `-mp`, `--model_path` | Model path | `~/.cache/modelscope/.../Qwen3-Reranker-0.6B` |
| `-mml`, `--max_model_length` | Max context length | 8192 |
| `-gmu`, `--gpu_memory_utilization` | GPU memory utilization | 0.2 |
| `-ss`, `--swap_space` | CPU swap space (GB) | 4 |
| `-eg`, `--eager_mode` | Eager mode | False |

</details>
