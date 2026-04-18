import subprocess
import time
import os
import signal
import json
import requests
import statistics

def benchmark(model_path, kv_bits, kv_quant_scheme, port=8080):
    print(f"--- Benchmarking: {model_path} | KV Bits: {kv_bits} | Scheme: {kv_quant_scheme} ---")
    
    cmd = [
        "uv", "run", "python", "-m", "mlx_vlm.server",
        "--model", model_path,
        "--port", str(port),
        "--kv-bits", str(kv_bits),
        "--kv-quant-scheme", kv_quant_scheme
    ]
    
    # Start the server in a new process group
    process = subprocess.Popen(cmd, preexec_fn=os.setsid)
    
    # Wait for server to be ready
    max_retries = 60
    ready = False
    for i in range(max_retries):
        try:
            response = requests.get(f"http://localhost:{port}/v1/models")
            if response.status_code == 200:
                ready = True
                break
        except:
            pass
        time.sleep(2)
    
    if not ready:
        print("Server failed to start.")
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        return None

    # Warmup
    print("Warming up...")
    try:
        requests.post(f"http://localhost:{port}/v1/chat/completions", json={
            "model": model_path,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 10
        }, timeout=30)
    except Exception as e:
        print(f"Warmup failed: {e}")

    # Actual benchmark
    print("Running inference test...")
    latencies = []
    tokens_per_sec = []
    
    payload = {
        "model": model_path,
        "messages": [{"role": "user", "content": "Write a long essay about the history of artificial intelligence."}],
        "max_tokens": 256,
        "stream": False
    }

    try:
        start_time = time.time()
        response = requests.post(f"http://localhost:{port}/v1/chat/completions", json=payload, timeout=120)
        end_time = time.time()
        
        if response.status_code == 200:
            data = response.json()
            total_time = end_time - start_time
            content = data['choices'][0]['message']['content']
            # Rough token count (words * 1.3 or just split)
            token_count = len(content.split()) * 1.3 
            # Better: use usage from response if available
            if 'usage' in data:
                token_count = data['usage']['completion_tokens']
            else:
                token_count = len(content.split()) * 1.3

            tps = token_count / total_time
            print(f"Tokens: {token_count:.1f}, Time: {total_time:.2f}s, TPS: {tps:.2f}")
            
            # Return results
            return {
                "model": model_path,
                "kv_bits": kv_bits,
                "kv_scheme": kv_quant_scheme,
                "tps": tps,
                "latency": total_time,
                "tokens": token_count
            }
        else:
            print(f"Request failed: {response.text}")
    except Exception as e:
        print(f"Inference failed: {e}")
    finally:
        # Cleanup
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        time.sleep(1)

    return None

if __name__ == "__main__":
    results = []
    # Test configurations
    # 1. Original
    results.append(benchmark("models/gemma4-26b-a4b-8bit", 3.5, "turboquant"))
    
    # 2. Try higher KV bits (if memory allows)
    results.append(benchmark("models/gemma4-26b-a4b-8bit", 4.0, "turboquant"))
    
    # 3. Try 4-bit model (should be much faster)
    results.append(benchmark("models/gemma4-26b-a4b-4bit", 4.0, "turboquant"))

    # 4. Try 31b model (if it fits)
    results.append(benchmark("models/gemma4-31b-4bit", 4.0, "turboquant"))

    # Filter out None results
    valid_results = [r for r in results if r is not None]
    
    if valid_results:
        print("\n--- FINAL RESULTS ---")
        for r in valid_results:
            print(f"Model: {r['model']} | KV: {r['kv_bits']} | TPS: {r['tps']:.2f}")
    else:
        print("No successful benchmarks.")
