#!/usr/bin/env python3
"""
Concurrent benchmark for GLM-4.7 inference
Tests throughput with multiple parallel requests
"""
import asyncio
import httpx
import time
import json
from statistics import mean, median, stdev

# Configuration
BASE_URL = "http://nvl72d012-T03:8000"
MODEL = "glm-4.7"
PROMPT = "Count from 1 to 20"
MAX_TOKENS = 100
CONCURRENT_REQUESTS = [1, 5, 10, 20, 50]  # Test different concurrency levels

async def send_request(client: httpx.AsyncClient, request_id: int):
    """Send a single request and measure latency"""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS
    }
    
    start = time.time()
    try:
        response = await client.post(
            f"{BASE_URL}/v1/chat/completions",
            json=payload,
            timeout=300.0  # 5 min timeout
        )
        response.raise_for_status()
        latency = time.time() - start
        
        data = response.json()
        usage = data.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)
        
        return {
            "success": True,
            "latency": latency,
            "completion_tokens": completion_tokens,
            "request_id": request_id
        }
    except Exception as e:
        return {
            "success": False,
            "latency": time.time() - start,
            "error": str(e),
            "request_id": request_id
        }

async def run_concurrent_benchmark(concurrency: int, num_requests: int = None):
    """Run benchmark with specified concurrency level"""
    if num_requests is None:
        num_requests = concurrency * 10  # 10 batches per concurrency level
    
    print(f"\n{'='*60}")
    print(f"Testing with {concurrency} concurrent requests ({num_requests} total)")
    print(f"{'='*60}")
    
    # Warmup request
    if concurrency == 1:
        print("Sending warmup request...")
        async with httpx.AsyncClient() as client:
            await send_request(client, 0)
        print("Warmup complete\n")
    
    async with httpx.AsyncClient() as client:
        start_time = time.time()
        
        # Send all requests concurrently
        tasks = [send_request(client, i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
    
    # Analyze results
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    if not successful:
        print("❌ All requests failed!")
        for r in failed:
            print(f"  Request {r['request_id']}: {r['error']}")
        return
    
    latencies = [r["latency"] for r in successful]
    total_tokens = sum(r["completion_tokens"] for r in successful)
    
    # Calculate metrics
    throughput = len(successful) / total_time
    tokens_per_sec = total_tokens / total_time
    
    print(f"✅ Results:")
    print(f"  Successful: {len(successful)}/{num_requests} requests")
    if failed:
        print(f"  Failed: {len(failed)} requests")
    print(f"\n📊 Latency (seconds):")
    print(f"  Min:    {min(latencies):.2f}s")
    print(f"  Median: {median(latencies):.2f}s")
    print(f"  Mean:   {mean(latencies):.2f}s")
    print(f"  Max:    {max(latencies):.2f}s")
    if len(latencies) > 1:
        print(f"  StdDev: {stdev(latencies):.2f}s")
    print(f"\n🚀 Throughput:")
    print(f"  Requests/sec: {throughput:.2f}")
    print(f"  Tokens/sec:   {tokens_per_sec:.2f}")
    print(f"  Total time:   {total_time:.2f}s")

async def main():
    """Run benchmark suite"""
    print("="*60)
    print("GLM-4.7 Concurrent Benchmark")
    print(f"Endpoint: {BASE_URL}")
    print(f"Model: {MODEL}")
    print(f"Max tokens: {MAX_TOKENS}")
    print("="*60)
    
    for concurrency in CONCURRENT_REQUESTS:
        await run_concurrent_benchmark(concurrency)
        await asyncio.sleep(2)  # Brief pause between tests
    
    print(f"\n{'='*60}")
    print("Benchmark complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    asyncio.run(main())
