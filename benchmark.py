import subprocess
import time
import json
import requests
import os
import signal
import statistics

def benchmark(model_path, kv_bits, kv_quant_scheme, port=8080):
    cmd = [
        "uv", "run", "python", "-m", "mlx_vlm.server",
        "--model", model_path,
        "--port", str(port),
        "--kv-bits", str(kv_bits),
        "--kv-quant-scheme", kv_quant_scheme
    ]
    
    print(f"Starting server: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, preexec_fn=os.setsid)
    
    # Wait for server to be ready
    max_retries = 60
    for i in range(max_retries):
        try:
            response = requests.get(f"http://localhost:{port}/v1/models")
            if response.status_code == 200:
                print("Server is up!")
                break
        except:
            pass
        time.sleep(2)
    else:
        print("Server failed to start.")
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
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

    # Benchmark
    print("Running benchmark...")
    latencies = []
    tokens_per_sec = []
    
    prompt = "Explain the concept of quantum entanglement in detail."
    
    for _ in range(3):
        start_time = time.time()
        try:
            response = requests.post(f"http://localhost:{port}/v1/chat/completions", json={
                "model": model_path,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 128
            }, timeout=60)
            
            end_time = time.time()
            data = response.json()
            
            duration = end_time - start_time
            content = data['choices'][0]['message']['content']
            # Rough token count by words/chars if not provided
            token_count = len(content.split()) * 1.3 # Approximation
            
            tps = token_count / duration
            latencies.append(duration)
            tokens_per_sec.append(tps)
            print(f"Iteration done. TPS: {tps:.2f}")
            
        except Exception as e:
            print(f"Request failed: {e}")

    # Cleanup
    print("Shutting down server...")
    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    time.sleep(2)

    if not tokens_per_sec:
        return None

    return {
        "avg_tps": statistics.mean(tokens_per_sec),
        "avg_latency": statistics.mean(latencies)
    }

if __name__ == "__main__":
    import sys
    model = sys.argv[1]
    bits = sys.argv[2]
    scheme = sys.argv[3]
    
    res = benchmark(model, bits, scheme)
    if res:
        print(json.dumps(res))
