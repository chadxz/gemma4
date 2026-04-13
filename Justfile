set shell := ["zsh", "-eu", "-o", "pipefail", "-c"]

mlx_root := "models"
turbo_opts := "--kv-bits 3.5 --kv-quant-scheme turboquant"

default:
    @just --list

sync:
    uv sync

server model_path:
    uv run python -m mlx_vlm server --model "{{ model_path }}" --port 8080 {{ turbo_opts }}

server-26b-a4b-4bit:
    uv run python -m mlx_vlm server --model "{{ mlx_root }}/gemma4-26b-a4b-4bit" --port 8080 {{ turbo_opts }}

server-26b-a4b-8bit:
    uv run python -m mlx_vlm server --model "{{ mlx_root }}/gemma4-26b-a4b-8bit" --port 8080 {{ turbo_opts }}

server-31b-4bit:
    uv run python -m mlx_vlm server --model "{{ mlx_root }}/gemma4-31b-4bit" --port 8080 {{ turbo_opts }}

server-31b-8bit:
    uv run python -m mlx_vlm server --model "{{ mlx_root }}/gemma4-31b-8bit" --port 8080 {{ turbo_opts }}

chat model_path:
    uv run python -m mlx_vlm.chat --model "{{ model_path }}" {{ turbo_opts }}

chat-26b-a4b-4bit:
    uv run python -m mlx_vlm.chat --model "{{ mlx_root }}/gemma4-26b-a4b-4bit" {{ turbo_opts }}

chat-26b-a4b-8bit:
    uv run python -m mlx_vlm.chat --model "{{ mlx_root }}/gemma4-26b-a4b-8bit" {{ turbo_opts }}

chat-31b-4bit:
    uv run python -m mlx_vlm.chat --model "{{ mlx_root }}/gemma4-31b-4bit" {{ turbo_opts }}

chat-31b-8bit:
    uv run python -m mlx_vlm.chat --model "{{ mlx_root }}/gemma4-31b-8bit" {{ turbo_opts }}
