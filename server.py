#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import time
from typing import (Any, Callable, Deque, Dict, List, Iterable, Iterator, Sequence, Set, Tuple, Type, Union)
import math
import uuid
import asyncio
import subprocess
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone, UTC

sys.path.append(os.path.realpath('site-packages'))

import fastapi
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi_offline import FastAPIOffline
from pydantic import BaseModel
import uvicorn
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoModel, is_torch_npu_available
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel
from vllm.inputs.data import TokensPrompt
from sentence_transformers import CrossEncoder, SentenceTransformer

import sys_util as sutil
from log_util import logger, config_logger, get_uvicorn_logging_config

thread_executor = ThreadPoolExecutor(max_workers=1)


class Qwen3Rerankervllm(CrossEncoder):
    def __init__(self, model_name_or_path, instruction="Given the user query, retrieval the relevant passages",
                 max_model_length: int = 8192,
                 gpu_memory_utilization: float = 0.5):
        logger.info(f'gpu_count={gpu_device_count}')
        self.instruction = instruction
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.suffix = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.max_model_length=max_model_length
        self.suffix_tokens: list[int] = self.tokenizer.encode(self.suffix, add_special_tokens=False)
        logger.info(f'suffix_tokens={self.suffix_tokens}')
        self.remain_capacity = self.max_model_length - len(self.suffix_tokens) - 2
        self.true_token = self.tokenizer("yes", add_special_tokens=False).input_ids[0]
        self.false_token = self.tokenizer("no", add_special_tokens=False).input_ids[0]
        self.sampling_params = SamplingParams(temperature=0,
            top_p=0.95,
            max_tokens=1,
            logprobs=20,
            allowed_token_ids=[self.true_token,self.false_token],
        )
        self.llm = LLM(model=model_name_or_path,
            tensor_parallel_size=gpu_device_count,
            max_model_len=self.max_model_length,
            enable_prefix_caching=True,
            # distributed_executor_backend='ray',
            gpu_memory_utilization=gpu_memory_utilization,
            swap_space=swap_space,
            # block_size=16,
            enforce_eager=eager_mode,
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.remain_capacity - 200,
            chunk_overlap=200,
            length_function=lambda x: len(self.tokenizer.encode(x, add_special_tokens=False)),
            separators=[
                "\n\n", "\n",     # 段落/换行
                "。", "！", "？",   # 中文句号/感叹号/问号
                "；", "。", "…",   # 分号、句号、省略号
                " ",              # 空格
                "",               # 字符级 fallback
            ],
        )

    def format_instruction(self, instruction, query, doc) -> list[dict[str,str]]:
        messages = [
            {"role": "system", "content": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."},
            {"role": "user", "content": f"<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {doc}"}
        ]
        return messages

    def compute_scores(self, uid:str, pairs: list[tuple[str,str]]
        ) -> tuple[list[float], list[int], dict[int, list[int]], dict[int, list[float]]]:
        messages: list[dict[str,str]] = [self.format_instruction(self.instruction, query, doc) for query, doc in pairs]
        messages_tokens: list[list[int]] =  self.tokenizer.apply_chat_template(
            messages, tokenize=True, padding=False, truncation=False, add_generation_prompt=False, enable_thinking=False
        )
        token_counts = [len(ele) for ele in messages_tokens]
        logger.info(f'apply_chat_template id={uid} token_counts={token_counts}, max={max(token_counts)}')
        long_token_parts_counts: dict[int, list[int]] = {}
        long_token_parts_scores: dict[int, list[float]] = {}

        long_text_index: list[int] = [index for index, count in enumerate(token_counts) if count > self.remain_capacity]
        if long_text_index:
            extra_messages_split_count: dict[int, int] = {}
            extra_messages: list = []
            for index in long_text_index:
                query, long_doc = pairs[index]
                text_parts = self.text_splitter.split_text(long_doc)
                extra_messages.extend([self.format_instruction(self.instruction, query, doc) for doc in text_parts])
                extra_messages_split_count[index] = len(text_parts)

            extra_tokens = self.tokenizer.apply_chat_template(
                extra_messages, tokenize=True, padding=False, truncation=False, add_generation_prompt=False, enable_thinking=False
            )

            for index, split_count in extra_messages_split_count.items():
                extra_token = extra_tokens.pop(0)
                messages_tokens[index] = extra_token
                long_token_parts_counts[index] = [len(extra_token)]
                for i in range(split_count-1):
                    extra_token = extra_tokens.pop(0)
                    messages_tokens.append(extra_token)
                    long_token_parts_counts[index].append(len(extra_token))

        messages_tokens = [ele[:self.remain_capacity] + self.suffix_tokens for ele in messages_tokens]
        tokens_prompts = [TokensPrompt(prompt_token_ids=ele) for ele in messages_tokens]
        outputs = self.llm.generate(tokens_prompts, self.sampling_params, use_tqdm=False)
        scores = []
        for i in range(len(outputs)):
            final_logits = outputs[i].outputs[0].logprobs[-1]
            # token_count = len(outputs[i].outputs[0].token_ids)
            if self.true_token not in final_logits:
                true_logit = -10
            else:
                true_logit = final_logits[self.true_token].logprob
            if self.false_token not in final_logits:
                false_logit = -10
            else:
                false_logit = final_logits[self.false_token].logprob
            true_score = math.exp(true_logit)
            false_score = math.exp(false_logit)
            score = true_score / (true_score + false_score)
            score = round(score, 7)
            scores.append(score)

        if long_text_index:
            extra_scores = scores[len(pairs):]
            for index, split_count in extra_messages_split_count.items():
                first_score = scores[index]
                long_token_parts_scores[index] = [first_score]
                for i in range(split_count-1):
                    if extra_scores:
                        extra_score = extra_scores.pop(0)
                        long_token_parts_scores[index].append(extra_score)
                        if extra_score > first_score:
                            first_score = extra_score
                            scores[index] = first_score
                    else:
                        logger.error(f"no extra score found for index {index}, must be logical error")
            if extra_scores:
                logger.error(f"extra_scores not empty, must be logical error")
            scores = scores[:len(pairs)]

        return scores, token_counts, long_token_parts_counts, long_token_parts_scores

    def stop(self):
        destroy_model_parallel()


_model: Qwen3Rerankervllm = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # uvlogger = logging.getLogger('uvicorn.access')
    # uvlogger.setLevel(logging.INFO) # show http access log
    # uvlogger.setLevel(logging.WARNING) # does not show http access log, or uvicorn.run(app, log_level='warning')

    logger.info(f'starts, pid={os.getpid()}')
    global _model
    start_tick = time.perf_counter()
    _model = Qwen3Rerankervllm(model_name_or_path=model_path,
                               instruction="Retrieval document that can answer user's query",
                               max_model_length=max_model_length,
                               gpu_memory_utilization=gpu_memory_utilization,
                               )
    logger.info(f'load model cost={time.perf_counter() - start_tick:.3f}')

    # ---- above is start
    yield # yield between start and stop
    # ---- below is stop

    logger.info(f'call model.stop')
    _model.stop()
    logger.info(f'stops, pid={os.getpid()}')

_log_dir = 'logs'
os.makedirs(_log_dir, exist_ok=True)

app = FastAPIOffline(lifespan=lifespan)
app.mount("/log/file", StaticFiles(directory=_log_dir))

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start_time
    response.headers['X-Process-Time'] = f'{process_time:.6f}'
    return response

@app.exception_handler(Exception)
async def custom_exception_handler(request: Request, ex: Exception):
    logger.error(f'{request.method} {request.url}, {ex!r}\n{sutil.get_exception_stack()}')
    return JSONResponse(
        status_code=500 if not isinstance(ex, HTTPException) else ex.status_code,
        content={"code": -1, "message": f"An error occurred while processing your request: {ex!r}"},
    )

class RerankScore(BaseModel):
    index: int
    relevance_score: float
    token_count: int
    token_parts: list[int]|None = None
    score_parts: list[float]|None = None

class RerankReq(BaseModel):
    model: str = 'Qwen3-Reranker-0.6B'
    query: str = ''
    documents: List[str]

class RerankResp(BaseModel):
    id: str = ''
    results: list[RerankScore]

@app.get("/")
async def handle_root(req: Request):
    return 'Qwen3 reranker server is running!'

@app.get("/health")
async def handle_health(req: Request, nvidia_smi: int=0):
    result = {
        "status": "ok",
        "timestemp": f"{datetime.now()}",
        "model_name": model_path.split('/')[-1],
        "max_model_length": max_model_length,
        "gpu_device_count": gpu_device_count,
        "gpu_memory_utilization": gpu_memory_utilization,
        "CUDA_VISIBLE_DEVICES": os.environ.get('CUDA_VISIBLE_DEVICES', ''),
        "NVIDIA_VISIBLE_DEVICES": os.environ.get('NVIDIA_VISIBLE_DEVICES', ''),
    }
    if nvidia_smi:
        cmd_result = sutil.run_cmd(['nvidia-smi'], shell=False)
        result['nvidia_smi'] = cmd_result['stdout']
    return result

@app.post("/v1/rerank", response_model=RerankResp, response_model_exclude_none=True, summary="rerank documents", description='''
    rerank documents and return the corresponding revevance scores.''')
async def handle_rerank(req: Request, rerank_req: RerankReq):
    uid = uuid.uuid4().hex
    docs_len = [len(doc) for doc in rerank_req.documents]
    total_len = sum(docs_len)
    logger.info(f'client={req.client.host}:{req.client.port}, id={uid} query={rerank_req.query!r}'
                f', doc_count={len(rerank_req.documents)}, total_len={total_len}, docs_len={docs_len}')
    queries = [rerank_req.query] * len(rerank_req.documents)
    pairs = list(zip(queries, rerank_req.documents))
    start_tick = time.perf_counter()
    scores, token_counts, long_token_parts_counts, long_token_parts_scores = await asyncio.get_running_loop(
        ).run_in_executor(thread_executor, _model.compute_scores, uid, pairs)
    cost = time.perf_counter() - start_tick
    logger.info(f'id={uid}, cost={cost:.3f}, scores={scores}')
    rerank_scores = [RerankScore(index=index, relevance_score=score, token_count=token_counts[index])
                      for index, score in enumerate(scores)]
    for index, token_parts in long_token_parts_counts.items():
        rerank_scores[index].token_parts = token_parts
    for index, score_parts in long_token_parts_scores.items():
        rerank_scores[index].score_parts = score_parts
    return RerankResp(id=f'rerank-{uid}', results=rerank_scores)

def file_size_to_str(size_in_bytes: int) -> str:
    if size_in_bytes >= 1073741824:  # 1024**3
        return f'{size_in_bytes / 1073741824:.3f} GB'
    elif size_in_bytes >= 1048576:  # 1024**2
        return f'{size_in_bytes / 1048576:.3f} MB'
    elif size_in_bytes >= 1024:
        return f'{size_in_bytes / 1024:.3f} KB'
    elif size_in_bytes > 1:
        return f'{size_in_bytes} Bytes'
    else:
        return f'{size_in_bytes} Byte'

@app.get('/log/files.html', response_class=HTMLResponse, summary='日志文件列表页面', description='''''')
async def show_log_files(request: Request):
    html = '''
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Log Files</title>
<style>
body {
    background-color: rgb(240,240,240);
}
table {
    border-collapse: collapse;
    font-size: 18px;
}
td {
    border: 1px solid #ccc;
    padding: 4px;
}
</style>
</head>
<body>
<h1>Log Files</h1>
<table>
    <tr>
    <th>Name</th>
    <th>Size</th>
    <th>Modify Time</th>
    </tr>
%s
</table>
</body>
</html>
'''
    table_rows = []
    with os.scandir('logs') as it:
        for entry in it:
            if entry.is_file():
                stat = entry.stat()
                table_rows.append(f'''    <tr>
        <td><a href="file/{entry.name}">{entry.name}</a></td>
        <td>{file_size_to_str(stat.st_size)}</td>
        <td>{datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")}</td>
    </tr>''')
    if sys.platform != 'win32':
        table_rows.sort()
    return html % '\n'.join(table_rows)


if __name__ == '__main__':
    import argparse

    model_path = os.path.expanduser('~/.cache/modelscope/hub/models/Qwen/Qwen3-Reranker-0.6B')
    parser = argparse.ArgumentParser()
    parser.add_argument('-H', '--host', type=str, default='0.0.0.0', help='server host[0.0.0.0]')
    parser.add_argument('-p', '--port', type=int, default=9994, help='server port[9994]')
    parser.add_argument('-mp', '--model_path', type=str, default=model_path, help=f'model path[{model_path}]')
    parser.add_argument('-mml', '--max_model_length', type=int, default=8192, help='max model len[8192]')
    parser.add_argument('-gmu', '--gpu_memory_utilization', type=float, default=0.2, help='gpu memory utilization[0.2]')
    parser.add_argument('-ss', '--swap_space', type=int, default=4, help='cpu memory swap space[4]')
    parser.add_argument('-eg', '--eager_mode', default=False, action='store_true', help='eager mode[False]')
    '''
    pip install langchain
    CUDA_VISIBLE_DEVICES=0
    TOKENIZERS_PARALLELISM=(true | false)

    max_model_length 4096 needs 0.44 GB KV cache
    max_model_length 8192 needs 0.88 GB KV cache
    max_model_length 10240 needs 1.09 GB KV cache

    eager_mode=True(启动较快，大约30秒，但运行速度较慢):
    RTX3090 24G gpu_memory_utilization 0.12615(about 3428MB) can supoort max_model_length 4096 for 1 concurrency
    RTX3090 24G gpu_memory_utilization 0.1447(about 3876MB) can supoort max_model_length 8192 for 1 concurrency
    RTX3090 24G gpu_memory_utilization 0.154(about 4100MB) can supoort max_model_length 10240 for 1 concurrency

    eager_mode=False(启动变慢，大约100秒):
    RTX3090 24G gpu_memory_utilization 0.1448(about 4412 MB) can supoort max_model_length 8192 for 1 concurrency
    RTX3090 24G gpu_memory_utilization 0.182(about 5316 MB) can supoort max_model_length 8192 for 2 concurrency
    RTX3090 24G gpu_memory_utilization 0.258(about 7108 MB) can supoort max_model_length 8192 for 4 concurrency


    运行时输出，说明gpu_memory_utilization越大，支持更多并发
    max_num_batched_tokens=8192.
    [gpu_worker.py:276] Available KV cache memory: 3.52 GiB
    [kv_cache_utils.py:849] GPU KV cache size: 32,960 tokens
    [kv_cache_utils.py:853] Maximum concurrency for 8,192 tokens per request: 4.02x
    '''

    args = parser.parse_args()

    config_logger(logger, log_to_stdout=True, log_dir=_log_dir, log_file='vllm_qwen3_reranker.log')
    log_config = get_uvicorn_logging_config()

    host = args.host
    port = args.port
    model_path = args.model_path
    max_model_length = args.max_model_length
    gpu_memory_utilization = args.gpu_memory_utilization
    gpu_device_count=torch.cuda.device_count()
    swap_space = args.swap_space
    eager_mode = args.eager_mode

    logger.info(f'server starts at {host}:{port}, pid={os.getpid()}')
    logger.info(f'args={args}')

    uvicorn.run(app, lifespan='on', host=host, port=port, workers=1,
                log_level='info', log_config=log_config)
