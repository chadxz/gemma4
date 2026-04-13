set shell := ["zsh", "-eu", "-o", "pipefail", "-c"]

mlx_root := "models"
turbo_opts := "--kv-bits 3.5 --kv-quant-scheme turboquant"

default:
    @just --list

sync:
    uv sync
    ./scripts/apply-mlx-vlm-patch.sh

patch-mlx-vlm:
    ./scripts/apply-mlx-vlm-patch.sh

server model_path:
    ./scripts/apply-mlx-vlm-patch.sh
    uv run python -m mlx_vlm server --model "{{ model_path }}" --port 8080 {{ turbo_opts }}

server-26b-a4b-4bit:
    just server "{{ mlx_root }}/gemma4-26b-a4b-4bit"

server-26b-a4b-8bit:
    just server "{{ mlx_root }}/gemma4-26b-a4b-8bit"

server-31b-4bit:
    just server "{{ mlx_root }}/gemma4-31b-4bit"

server-31b-8bit:
    just server "{{ mlx_root }}/gemma4-31b-8bit"

chat model_path:
    ./scripts/apply-mlx-vlm-patch.sh
    uv run python -m mlx_vlm.chat --model "{{ model_path }}" {{ turbo_opts }}

chat-26b-a4b-4bit:
    just chat "{{ mlx_root }}/gemma4-26b-a4b-4bit"

chat-26b-a4b-8bit:
    just chat "{{ mlx_root }}/gemma4-26b-a4b-8bit"

chat-31b-4bit:
    just chat "{{ mlx_root }}/gemma4-31b-4bit"

chat-31b-8bit:
    just chat "{{ mlx_root }}/gemma4-31b-8bit"
