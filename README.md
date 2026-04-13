# Gemma 4 with `pi` and `mlx-vlm`

This repo gives you a small local workflow for running Gemma 4 MLX models that
were downloaded with LM Studio, then talking to them through
`pi` (`@mariozechner/pi-coding-agent`).

## Why `mlx-vlm`

LM Studio is still useful here, but mostly as the easiest way to download and
manage the MLX model files on disk.

We use `mlx-vlm` to actually run the model because it is the piece that gives
us:

- a working OpenAI-compatible local server for these Gemma 4 MLX models
- TurboQuant KV-cache support
- a clean way to point `pi` at a localhost endpoint

In practice, that means LM Studio is the downloader and local model manager,
while `mlx-vlm` is the runtime that `pi` talks to.

## Why the local patch

This repo carries a small local patch for `mlx-vlm` because its current
streaming tool-calling implementation can leak raw tool-call markup into the
assistant text stream.

That is especially noticeable in `pi`, where you end up seeing the model's raw
tool-call text in the transcript even though the tool also executes normally.

The patch keeps streaming enabled, but strips partial tool-call markup from the
visible assistant text and reports `finish_reason: "tool_calls"` when tools are
present. The result is a much cleaner `pi` experience without giving up
streaming.

It uses:

- `mise` for tool installation
- `uv` for the Python environment
- `just` for repeatable run commands
- `mlx-vlm` as the OpenAI-compatible local server

## Install `pi`

Install `pi` globally with npm:

```bash
npm install -g @mariozechner/pi-coding-agent
```

## Install project tools with `mise`

This project already has a local `mise.toml`.

Recommended: use `mise` shell integration so `uv`, `just`, and `python` are on
your `PATH` automatically while you are in this repo.

For `zsh`, add this to `~/.zshrc`:

```bash
eval "$(mise activate zsh)"
```

Then restart your shell and run:

```bash
cd ~/src/personal/gemma4
mise install
```

If you only want to enable it in the current shell, run:

```bash
eval "$(mise activate zsh)"
```

This README uses direct `just` and `uv` commands on purpose. That is the
preferred workflow here. `mise exec -- ...` still works, but it is more
awkward day to day.

## Sync the Python environment

From the repo root:

```bash
cd ~/src/personal/gemma4
just sync
```

`just sync` also reapplies the local `mlx-vlm` patch that keeps streamed tool
calls from leaking raw markup into `pi` output.

If you want to reapply that patch by itself:

```bash
cd ~/src/personal/gemma4
just patch-mlx-vlm
```

## Download the models in LM Studio

Download these MLX models in LM Studio:

- `mlx-community/gemma-4-26b-a4b-it-8bit`
- `mlx-community/gemma-4-26b-a4b-it-4bit`
- `mlx-community/gemma-4-31b-it-8bit`
- `mlx-community/gemma-4-31b-it-4bit`

This repo expects those models to exist under:

- `~/.lmstudio/models/mlx-community/...`

The local `models/` entries in this repo are symlinks that point at those LM
Studio downloads.

Create or refresh those symlinks with:

```bash
cd ~/src/personal/gemma4
mkdir -p models

ln -sfn ~/.lmstudio/models/mlx-community/gemma-4-26b-a4b-it-8bit \
  models/gemma4-26b-a4b-8bit
ln -sfn ~/.lmstudio/models/mlx-community/gemma-4-26b-a4b-it-4bit \
  models/gemma4-26b-a4b-4bit
ln -sfn ~/.lmstudio/models/mlx-community/gemma-4-31b-it-8bit \
  models/gemma4-31b-8bit
ln -sfn ~/.lmstudio/models/mlx-community/gemma-4-31b-it-4bit \
  models/gemma4-31b-4bit
```

## Configure `pi`

Use [pi.models.example.json](pi.models.example.json) as the `mlx-vlm`
provider block for `~/.pi/agent/models.json`.

The important details in that config are:

- short model ids like `models/gemma4-26b-a4b-8bit`
- `input: ["text", "image"]` for Gemma 4 vision support
- `compat.maxTokensField = "max_tokens"` so `pi` does not get cut off at the
  `mlx-vlm` default output limit

If you want these models in your scoped `pi` model list, add them to
`~/.pi/agent/settings.json` in `enabledModels` like this:

```json
[
  "mlx-vlm/models/gemma4-26b-a4b-8bit",
  "mlx-vlm/models/gemma4-26b-a4b-4bit",
  "mlx-vlm/models/gemma4-31b-8bit",
  "mlx-vlm/models/gemma4-31b-4bit"
]
```

## Run the server

Run these commands from the repo root so the relative `models/...` ids resolve
correctly.

Start one server at a time:

```bash
cd ~/src/personal/gemma4
just server-26b-a4b-8bit
```

Other variants:

```bash
just server-26b-a4b-4bit
just server-31b-8bit
just server-31b-4bit
```

These recipes run `mlx-vlm` with TurboQuant KV cache settings:

- `--kv-bits 3.5`
- `--kv-quant-scheme turboquant`

## Optional local chat

If you want to test the model without `pi` first:

```bash
cd ~/src/personal/gemma4
just chat-26b-a4b-8bit
```

Other variants:

```bash
just chat-26b-a4b-4bit
just chat-31b-8bit
just chat-31b-4bit
```

## Run `pi` against the local server

In another terminal:

```bash
pi --model "Gemma 4 26B A4B IT 8bit (mlx-vlm)"
```

Or choose one of the others:

```bash
pi --model "Gemma 4 26B A4B IT 4bit (mlx-vlm)"
pi --model "Gemma 4 31B IT 8bit (mlx-vlm)"
pi --model "Gemma 4 31B IT 4bit (mlx-vlm)"
```

## Notes

- You do not need LM Studio running once the model files are already downloaded.
- If `pi` shows stale model ids, restart `pi` after changing
  `~/.pi/agent/models.json` or `~/.pi/agent/settings.json`.
- If you move your LM Studio model directory, update the symlinks in `models/`.
