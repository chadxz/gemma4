#!/usr/bin/env python3

import argparse
import json
import time
import urllib.request


SYSTEM = (
    "You are a careful local coding assistant. Be concise, practical, and "
    "specific. When unsure, say what you would inspect next."
)


TURNS = [
    """
I am debugging a coding-agent harness in a monorepo. Here are the relevant
files.

File: apps/api/src/search.ts
```ts
import { fetchSearchResults } from "./vendor/brave";

export async function search(query: string, opts?: { freshness?: string }) {
  const response = await fetchSearchResults({
    q: query,
    freshness: opts?.freshness,
    count: 10,
  });

  return response.web?.results?.map((item: any) => ({
    title: item.title,
    url: item.url,
    snippet: item.description,
    age: item.age,
  })) ?? [];
}
```

File: apps/api/src/vendor/brave.ts
```ts
export async function fetchSearchResults(params: Record<string, string>) {
  const url = new URL("https://api.search.brave.com/res/v1/web/search");
  Object.entries(params).forEach(([key, value]) => {
    if (value) url.searchParams.set(key, value);
  });

  const response = await fetch(url, {
    headers: {
      Accept: "application/json",
      "X-Subscription-Token": process.env.BRAVE_API_KEY ?? "",
    },
  });

  if (!response.ok) {
    throw new Error(`Brave API failed: ${response.status}`);
  }

  return await response.json();
}
```

File: packages/agent/src/planner.ts
```ts
export function chooseSearchMode(input: string) {
  if (input.includes("today") || input.includes("latest")) {
    return "web";
  }
  if (input.includes("docs") || input.includes("reference")) {
    return "docs";
  }
  return "local";
}
```

Please identify the two biggest correctness risks and explain them briefly.
""".strip(),
    """
Thanks. Now assume the user reports that long coding sessions feel slower on
every turn even when they only add one small follow-up message.

Given the code above, plus this server command:

```bash
uv run python -m mlx_vlm.server \
  --model models/gemma4-26b-a4b-8bit \
  --port 8080 \
  --reuse-prompt-cache
```

What is the most likely performance explanation, and what would you measure
first?
""".strip(),
    """
Add this Python server excerpt to the context:

```py
def build_generation_kwargs(request, template_kwargs):
    return {
        "prefill_step_size": get_prefill_step_size(),
        "kv_bits": get_quantized_kv_bits(request.model),
        "kv_group_size": get_kv_group_size(),
        "kv_quant_scheme": get_kv_quant_scheme(),
        "max_kv_size": get_max_kv_size(request.model),
        "quantized_kv_start": get_quantized_kv_start(),
        **request.generation_kwargs(),
        **template_kwargs,
    }
```

Now give a short tuning recommendation with exact flag changes, optimized for a
developer laptop that values coding quality and responsiveness more than raw
throughput.
""".strip(),
    """
Final turn. Keep it tight.

Write a final recommendation in three bullets:
- default config
- lower-memory fallback
- one thing to avoid unless prompts get extremely long
""".strip(),
]


def call_chat(base_url: str, model: str, messages, max_tokens: int):
    payload = {
        "model": model,
        "stream": False,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    data = json.dumps(payload).encode()
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=600) as response:
        body = json.loads(response.read().decode())
    elapsed = time.perf_counter() - started
    return body, elapsed


def main():
    parser = argparse.ArgumentParser(description="Benchmark multi-turn coding chat.")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    messages = [{"role": "system", "content": SYSTEM}]
    turns = []

    for idx, user_turn in enumerate(TURNS, start=1):
        messages.append({"role": "user", "content": user_turn})
        body, elapsed = call_chat(
            base_url=args.base_url,
            model=args.model,
            messages=messages,
            max_tokens=args.max_tokens,
        )
        assistant = body["choices"][0]["message"]["content"]
        usage = body["usage"]
        turns.append(
            {
                "turn": idx,
                "wall_seconds": round(elapsed, 3),
                "input_tokens": usage["input_tokens"],
                "output_tokens": usage["output_tokens"],
                "prompt_tps": round(usage["prompt_tps"], 3),
                "generation_tps": round(usage["generation_tps"], 3),
                "peak_memory_gb": round(usage["peak_memory"], 3),
                "assistant_preview": assistant[:180],
            }
        )
        messages.append({"role": "assistant", "content": assistant})

    print(
        json.dumps(
            {
                "label": args.label,
                "model": args.model,
                "turns": turns,
            }
        )
    )


if __name__ == "__main__":
    main()
