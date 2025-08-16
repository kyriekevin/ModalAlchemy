---
title: uv Usage Notes
date: 2025-08-16T00:00:00.000Z
tags:
  - python_package
  - project_manager
category: codebase
author: zyz
---
# uv Usage Notes

## 📖 简介

An extremely fast Python package and project manager, written in Rust

## 📝 官方文档与资源

[GitHub](https://github.com/astral-sh/uv)
[Docs](https://docs.astral.sh/uv/)

## 📦 安装与环境

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 🚀 基础用法

### `uv` 文件说明

完整的uv项目结构如下

```
.
├── .venv
│   ├── bin
│   ├── lib
│   └── pyvenv.cfg
├── .python-version
├── README.md
├── main.py
├── pyproject.toml
└── uv.lock
```

- `pyproject.toml` file

```toml
[project]
name = "hello-world"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
dependencies = []
```

文件记录了元信息，可以使用`uv add`和`uv remove`来管理依赖

- `.python-version` file
  记录了项目默认的python版本

- `.venv` folder
  `uv` 创建的虚拟环境，独立于系统的python环境，项目依赖环境都添加于此

- `uv.lock` file
  `uv.lock` 是一个跨平台的锁文件，其中包含项目依赖项的精确信息。与用于指定项目总体需求的 `pyproject.toml` 不同，该锁文件包含项目环境中已安装的已解析版本。此文件应纳入版本控制，以便在不同机器上实现一致且可重复的安装。

### Pipeline

1. 创建项目 (Optional)

- 创建新python项目

  ```bash
  uv init xxx
  cd xxx
  ```

- 初始化已有项目

```bash
cd xxx
uv init
```

2. 创建激活虚拟环境

- 创建环境 (uv会自动寻找当前目录下的 `.python-version` 文件或使用可用的Python版本):

```bash
uv venv
uv venv --python 3.x # 指定python版本
```

- 激活环境

```bash
source .venv/bin/activate
```

3. 命令运行

- 未激活环境

```bash
uv run xxx.py
uv run bash xxx.sh
uv run <command>
```

- 激活环境

```bash
run xxx.py
bash xxx.sh
<command>
```

### 依赖管理

- 添加依赖，--dev、--group 或 --optional 标志可用于向替代字段添加依赖项

```bash
# uv add [dependency]
uv add requests
# 版本约束
uv add 'requests==2.31.0'
# Add a git dependency
uv add git+https://github.com/psf/requests
# Add all dependencies from `requirements.txt`.
uv add -r requirements.txt -c constraints.txt
```

- 对于项目使用requirements.txt管理的

```bash
uv add -r requirements.txt
```

- 工具安装 ruff、pre-commit等，以ruff为例

```bash
uv tool install ruff
uv tool run ruff format .
uvx ruff check . # uvx = uv tool run
```

- 移除依赖

```bash
uv remove xxx
```

### 常用命令

| command                          | description                                                                           |
| -------------------------------- | ------------------------------------------------------------------------------------- |
| uv python install/uninstall 3.xx | 安装卸载指定版本python                                                                |
| uv lock                          | 生成lock文件                                                                          |
| uv tree                          | 查看项目依赖树                                                                        |
| uv pip                           | 使用pip方式管理依赖，但不会同步到pyproject.toml，优先uv add。除了类似flash-attn特殊库 |

## 🌟 进阶用法

### flash-attn

```bash
uv add "flash-attn=xxx" --no-build-isolation # 推荐2.7.3
uv sync --no-build-isolation-package flash-attn
```
