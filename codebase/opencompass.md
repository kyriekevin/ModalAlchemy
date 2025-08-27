---
title: opencompass Usage Notes
date: 2025-08-25T00:00:00.000Z
tags:
  - eval
  - vllm
category: codebase
author: zyz
---
# opencompass Usage Notes

## 📖 简介

OpenCompass is an LLM evaluation platform, supporting a wide range of models (Llama3, Mistral, InternLM2,GPT-4,LLaMa2, Qwen,GLM, Claude, etc) over 100+ datasets.

## 📝 官方文档与资源

[GitHub](https://github.com/open-compass/opencompass)
[Docs](https://opencompass.readthedocs.io/en/latest/)

## 📦 安装与环境

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.10
uv init
uv add "opencompass[vllm]" langdetect "math-verify[antlr4_11_0]"
uv add "git+https://github.com/open-compass/human-eval.git#egg=human-eval"
```

## 🚀 基础用法

### Config

#### 运行 Config

```python
from mmengine.config import read_base
from opencompass.partitioners import NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask

with read_base():
    from .datasets.IFEval.IFEval_gen import ifeval_datasets
    from .datasets.mmlu_pro.mmlu_pro_gen import mmlu_pro_datasets
    from .datasets.math.math_prm800k_500_gen import math_datasets
    from .datasets.humaneval.humaneval_gen import humaneval_datasets
    from .datasets.gpqa.gpqa_gen import gpqa_datasets
    from .datasets.aime2024.aime2024_gen import aime2024_datasets
    from .datasets.mbpp.mbpp_gen import mbpp_datasets
    from .models.ttls.qwen import models

datasets = [*mmlu_pro_datasets, *ifeval_datasets, *math_datasets, *humaneval_datasets, *gpqa_datasets, *aime2024_datasets, *mbpp_datasets]
models
```

指定需要评测的模型和评测集

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
uv run nohup python3 run.py opencompass/configs/ttls.py --max-num-workers 8
```

以上为8卡vllm评测示例，相应可以减少卡数，调整`CUDA_VISIBLE_DEVICES` 和 `--max-num-workers` 即可

#### Model

```python
from opencompass.models import VLLMwithChatTemplate
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr='qwen2.5-7b-instruct-vllm',
        path='',
        model_kwargs=dict(tensor_parallel_size=1),
        max_out_len=4096,
        batch_size=16,
        generation_kwargs=dict(temperature=0),
        run_cfg=dict(num_gpus=1),
    )
]
```

以 `Qwen2.5-7B-Instruct`为例,在运行config中加载对应模型。对于`Qwen3`涉及Think开关的请参考进阶用法部分。

#### Dataset

当前选取以下数据集作为Benchmark

| Dataset   | Domain                | Desc                                                                |
| --------- | --------------------- | ------------------------------------------------------------------- |
| MMLU-Pro  | Knowledge             | 高难度多学科选择题，选项更多，专测大模型知识与推理能力。            |
| GPQA      | Knowledge             | 研究生水平、难以搜索的理科多选题，考查深度理解与推理。              |
| AIME 2024 | Math                  | 2024年AIME竞赛原题，衡量模型高中竞赛数学解题能力。                  |
| Math 500  | Math                  | 500道高难度数学竞赛题，主流数学推理评测集。                         |
| Humaneval | Code                  | 164道Python编程题，主流代码生成能力评测集。                         |
| ~~MBPP~~  | ~~Code~~              | ~~974道基础~~~~Python~~~~编程题，适合初学者，常用于代码能力评测。~~ |
| IFEval    | Instruction Following | 主流指令跟随能力评测集，覆盖多任务。                                |

> MBPP测试精度差异大难以对齐，不采用

### Output

推理结果默认存储在`outputs/default`下

#### Prediction

可以在 `outputs/default/<date>/predictions/<model abbr>`下看到模型具体的生成内容
可以查看其中`prediction`来确认模型生成是否异常

#### Summary

- Qwen2.5-7B-Instruct

  - 8卡`vllm`耗时：15min44s
    - 评测结果
      ![image.png](https://raw.githubusercontent.com/kyriekevin/img_auto/main/Obsidian/202508251854378.png)
      ![image.png](https://raw.githubusercontent.com/kyriekevin/img_auto/main/Obsidian/202508251948585.png)
  - 精度对比

    | Datasets  | Opencompass | Official | Deviation  |
    | --------- | ----------- | -------- | ---------- |
    | MMLU-Pro  | 57.64       | 56.3     | +1.34      |
    | GPQA      | 34.34       | 36.4     | -2.06      |
    | HumanEval | 85.37       | 84.8     | +0.57      |
    | Math 500  | 76.2        | 75.5     | +0.7       |
    | ~~MBPP~~  | ~~58.99~~   | ~~79.2~~ | ~~-20.21~~ |
    | IFEval    | 71.72       | 71.2     | +0.52      |

    - vllm本身推理有精度波动，且shot数可能不一致会导致精度无法完全对齐

## 🌟 进阶用法

### Think开关

```Python
from opencompass.models import VLLMwithChatTemplate
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr='qwen3-8b-wo-think-vllm',
        path='',
        model_kwargs=dict(tensor_parallel_size=1),
        max_seq_len=32768,
        max_out_len=32000,
        batch_size=16,
        run_cfg=dict(num_gpus=1),
        chat_template_kwargs=dict(enable_thinking=False),
        generation_kwargs=dict(temperature=0.7, top_p=0.8, top_k=20),
        pred_postprocessor=dict(type=extract_non_reasoning_content)
    ),
    dict(
        type=VLLMwithChatTemplate,
        abbr='qwen3-8b-w-think-vllm',
        path='',
        model_kwargs=dict(tensor_parallel_size=1),
        max_seq_len=32768,
        max_out_len=32000,
        batch_size=16,
        run_cfg=dict(num_gpus=1),
        chat_template_kwargs=dict(enable_thinking=True),
        generation_kwargs=dict(temperature=0.6, top_p=0.95, top_k=20),
        pred_postprocessor=dict(type=extract_non_reasoning_content)
    ),
]
```

> 注：通过`chat_template_kwargs`字段控制是否开启think，`generation_kwargs`参考官方`Best Practices`

- Qwen3 8B (Non-thinking)

  - 8卡`vllm`耗时：1h58min18s
  - 评测结果
    ![image.png](https://raw.githubusercontent.com/kyriekevin/img_auto/main/Obsidian/202508252219885.png)
    ![image.png](https://raw.githubusercontent.com/kyriekevin/img_auto/main/Obsidian/202508252219273.png)
  - 精度对比

    | Datasets  | Opencompass | Official | Deviation |
    | --------- | ----------- | -------- | --------- |
    | GPQA      | 43.94       | 39.3     | +4.64     |
    | IFEval    | 82.81       | 83       | -0.19     |
    | Math 500  | 86.2        | 87.4     | -1.2      |
    | AIME 2024 | 26.67       | 29.1     | -2.43     |

- Qwen3 8B (thinking)

  - 8卡`vllm`耗时：3h19min26s
  - 评测结果
    ![image.png](https://raw.githubusercontent.com/kyriekevin/img_auto/main/Obsidian/202508252356183.png)
    ![image.png](https://raw.githubusercontent.com/kyriekevin/img_auto/main/Obsidian/202508252356309.png)

  - 精度对比

    | Datasets  | Opencompass | Official | Deviation |
    | --------- | ----------- | -------- | --------- |
    | GPQA      | 54.55       | 62       | -7.45     |
    | IFEval    | 82.44       | 85       | -2.56     |
    | Math 500  | 96.8        | 97.4     | -0.6      |
    | AIME 2024 | 80          | 76       | +4        |
