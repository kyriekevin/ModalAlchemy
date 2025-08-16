---
title: vLLM Usage Notes
date: 2025-08-16T00:00:00.000Z
tags:
  - codebase
  - vllm
  - inference
category: codebase
author: zyz
---
# vLLM Usage Notes

## 📖 简介

A high-throughput and memory-efficient inference and serving engine for LLMs

## 📝 官方文档与资源

- [GitHub](https://github.com/vllm-project/vllm)
- [Docs](https://docs.vllm.ai/en/latest/)

## 📦 安装与环境

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.10
uv init
uv add torch torchvision torchaudio "vllm>=0.8.3" "transformers==4.51.3" qwen_vl_utils setuptools
uv add "flash-attn==2.7.3" --no-build-isolation
```

## 🚀 基础用法

以 `Qwen` 系列为例展示 `LLM` 和 `MLLM` 推理方式

### LLM

`Qwen3 8B` model inference code

```python
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MODEL_PATH = ""

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

sampling_params = SamplingParams(
    temperature=0.6, top_p=0.95, top_k=20, max_tokens=32768
)

llm = LLM(model=MODEL_PATH, trust_remote_code=True)

prompt = ""
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,  # Set to False to strictly disable thinking
)

outputs = llm.generate([text], sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

> 需要替换内容：超参数(包括是否开启thinking)、模型路径和messages内容

### MLLM

`Qwen2.5 vl 7B` model inference code

```python
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

MODEL_PATH = ""

llm = LLM(
    model=MODEL_PATH,
    limit_mm_per_prompt={"image": 10},
)

sampling_params = SamplingParams(
    temperature=0.1,
    top_p=0.001,
    repetition_penalty=1.05,
    max_tokens=256,
    stop_token_ids=[],
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "",
                "min_pixels": 224 * 224,
                "max_pixels": 1280 * 28 * 28,
            },
            {"type": "text", "text": "Describe this image."},
        ],
    },
]

processor = AutoProcessor.from_pretrained(MODEL_PATH)
prompt = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
image_inputs, video_inputs = process_vision_info(messages)

mm_data = {}
if image_inputs is not None:
    mm_data["image"] = image_inputs
if video_inputs is not None:
    mm_data["video"] = video_inputs

llm_inputs = {
    "prompt": prompt,
    "multi_modal_data": mm_data,
}

outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
generated_text = outputs[0].outputs[0].text

print(generated_text)
```

> 需要替换内容：超参数、模型路径和messages内容

## 🌟 进阶用法

### 多卡Batch推理

```python
import multiprocessing as mp
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
from pathlib import Path
import json
import os

def batch_generator(data_generator, batch_size):
    batch = []
    for item in data_generator:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def prepare_batch_inputs(records, processor, num_samples=8):
    batch_inputs = []
    batch_metadata = []

    for record_idx, record in enumerate(records):
        image_urls = record['images']
        placeholders = [{"type": "image", "image": url} for url in image_urls]

        messages = [
            {
                "role": "user",
                "content": [
                    *placeholders,
                    {"type": "text", "text": record['messages'][0]['content'].replace("<image>", "")}
                ]
            }
        ]

        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = process_vision_info(messages)

        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs

        llm_inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
        }

        for sample_id in range(num_samples):
            batch_inputs.append(llm_inputs)
            batch_metadata.append({
                "prompt": record['messages'][0]['content'],
                "label": record['messages'][1]['content'],
                "sample_id": sample_id,
                "original_index": record.get('index', -1),
                "record_idx": record_idx
            })

    return batch_inputs, batch_metadata

def process_large_batch(records, processor, llm, sampling_params, num_samples=8):
    batch_inputs, batch_metadata = prepare_batch_inputs(records, processor, num_samples)

    print(f"Processing {len(records)} records with {num_samples} samples each = {len(batch_inputs)} total inferences")

    outputs = llm.generate(batch_inputs, sampling_params=sampling_params)

    results = []
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        metadata = batch_metadata[i]

        result = {
            "prompt": metadata["prompt"],
            "label": metadata["label"],
            "predict": generated_text,
            "sample_id": metadata["sample_id"],
            "original_index": metadata["original_index"]
        }
        results.append(result)

    return results

def worker_process(rank, world_size, data_chunk, model_path, output_file, num_samples=8):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

    model_len = 4096

    llm = LLM(
        model=model_path,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=model_len,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.95,
        max_num_batched_tokens=8192,
        dtype="bfloat16",
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        seed=None,
        max_tokens=model_len,
        stop=["</answer>", "<think>"]
    )

    processor = AutoProcessor.from_pretrained(model_path)

    results = []

    batch_size = 16

    for batch_idx, batch in enumerate(batch_generator(data_chunk, batch_size)):
        print(f"Rank {rank}: Processing batch {batch_idx}, {len(batch)} records -> {len(batch) * num_samples} inferences")

        batch_results = process_large_batch(batch, processor, llm, sampling_params, num_samples)
        results.extend(batch_results)

        print(f"Rank {rank}: Completed batch {batch_idx}, total results: {len(results)}")

    with open(f"{output_file}_rank_{rank}.jsonl", "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"Rank {rank} finished processing {len(data_chunk)} items, generated {len(results)} samples")

def main():
    folder_path = ""
    MODEL_PATH = ""

    with open(folder_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    for i, record in enumerate(data):
        record['index'] = i

    world_size = 8
    num_samples = 8
    chunk_size = len(data) // world_size
    data_chunks = []

    for i in range(world_size):
        start_idx = i * chunk_size
        if i == world_size - 1:
            end_idx = len(data)
        else:
            end_idx = (i + 1) * chunk_size
        data_chunks.append(data[start_idx:end_idx])

    print(f"Total data: {len(data)}, will generate {len(data) * num_samples} samples")
    print(f"Each GPU will process ~{len(data) // world_size} records -> ~{(len(data) // world_size) * num_samples} inferences")

    processes = []
    for rank in range(world_size):
        p = mp.Process(
            target=worker_process,
            args=(rank, world_size, data_chunks[rank], MODEL_PATH, "result", num_samples)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    all_results = []
    for rank in range(world_size):
        with open(f"result_rank_{rank}.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                all_results.append(json.loads(line))

    output_path = ""
    with open(output_path, "w", encoding="utf-8") as f:
        for result in all_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"Total processed: {len(all_results)} samples from {len(data)} original records")

if __name__ == "__main__":
    main()
```

> 如果出现nvidia-smi显示memory高 gpu利用率为0，可能是参数问题导致卡住。可以调整batch_size、num_samples、gpu_memory_utilization、max_num_batched_tokens等参数

### 72B Model

`Qwen2.5 vl 72B` model inference code

```python
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
from pathlib import Path
import json

llm = LLM(
    model=MODEL_PATH,
    enforce_eager=True,
    tensor_parallel_size=4, # 核心是tp并行
    gpu_memory_utilization=0.95,
    dtype="bfloat16"
)
```

### 截断处理

```python
llm = LLM(
    model=MODEL_PATH,
    enforce_eager=True,
    tensor_parallel_size=1,
    max_model_len=16384,
)

sampling_params = SamplingParams(
    temperature=0,
    seed=42,
    max_tokens=256,
    stop=["</answer>"]
)
```

指定截断的word，以PAR为例格式是 `<answer> xxx </answer>`，设置stop中包括`</answer>`则不会继续输出`<think> xxx </think>`内容

> 注: PAR指(Prompt-Answer-Reason)格式

### Prefix cache

```python
llm = LLM(
    model=MODEL_PATH,
    enforce_eager=True,
    tensor_parallel_size=1,
    max_model_len=16384,
    enable_prefix_caching=True,
)
```

设置 `prefix_caching` 来对相同prompt模板做计算优化

### Yarn

```python
rope_scaling = {
    "rope_type": "yarn",
    "factor": 1.0,
    "original_max_position_embeddings": 128000
}

llm = LLM(
    model=model_path,
    enforce_eager=True,
    tensor_parallel_size=1,
    max_model_len=model_len,
    gpu_memory_utilization=0.95,
    dtype="bfloat16",
    rope_scaling=rope_scaling
)
```

对于使用 `Yarn` 技术训练的超长文本模型，需要设置 `rope` 配置，其中 `factor` 是扩展的倍数对于原始长度而言，`original_max_position_embeddings` 是通过查看 `config.json` 查看原始模型最大长度

### gpt-oss

1. 下载模型

```bash
sudo apt install aria2
wget https://hf-mirror.com/hfd/hfd.sh
chmod a+x hfd.sh

hfd.sh "openai/gpt-oss-20b"
```

2. 安装vllm

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12
uv init
uv add openai-harmony
uv pip install --pre vllm==0.10.1+gptoss \
    --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
    --index-strategy unsafe-best-match

# 添加 Debian 12 的源来获取新版 glibc
echo "deb http://deb.debian.org/debian bookworm main" | sudo tee /etc/apt/sources.list.d/bookworm.list

# 更新包列表
sudo apt update -y

# 安装新版 glibc
sudo apt install -t bookworm libc6 -y

# 设置 LD_LIBRARY_PATH
export LD_LIBRARY_PATH=~/glibc-new/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

ldd --version
```

3. 推理

[对话格式](https://cookbook.openai.com/articles/openai-harmony)

```python
import json
from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    Conversation,
    Message,
    Role,
    SystemContent,
    DeveloperContent,
    ReasoningEffort,
)

from vllm import LLM, SamplingParams

encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

system_message = (
    SystemContent.new()
        .with_model_identity(
            "You are ChatGPT, a large language model trained by OpenAI."
        )
        .with_reasoning_effort(ReasoningEffort.LOW)
        .with_conversation_start_date("2025-08-06")
        .with_knowledge_cutoff("2024-06")
        .with_required_channels(["analysis", "commentary", "final"])
)

convo = Conversation.from_messages(
    [
        Message.from_role_and_content(Role.SYSTEM, system_message),
        Message.from_role_and_content(Role.USER, "Please introduce yourself"),
    ]
)

prefill_ids = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)

stop_token_ids = encoding.stop_tokens_for_assistant_actions()

llm = LLM(
    model="/opt/tiger/vlm/gpt-oss-20b",
    trust_remote_code=True,
)

sampling = SamplingParams(
    max_tokens=4096,
    temperature=1,
    stop_token_ids=stop_token_ids,
)

outputs = llm.generate(
    prompt_token_ids=[prefill_ids],
    sampling_params=sampling,
)

gen = outputs[0].outputs[0]
text = gen.text
output_tokens = gen.token_ids

entries = encoding.parse_messages_from_completion_tokens(output_tokens, Role.ASSISTANT)

for message in entries:
    print(f"{json.dumps(message.to_dict())}")
```

![image.png](https://raw.githubusercontent.com/kyriekevin/img_auto/main/Obsidian/202508161640523.png)
