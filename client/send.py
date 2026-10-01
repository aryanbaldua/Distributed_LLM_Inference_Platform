"""Send completions to the master and report which worker served each one.

Concurrency is the point: one request at a time cannot show load being spread,
because each finishes before the next is dispatched.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from collections import Counter

import httpx


async def send_one(http, url, model, prompt, semaphore, index):
    async with semaphore:
        started = time.perf_counter()
        try:
            response = await http.post(
                f"{url}/v1/chat/completions",
                json={"model": model, "messages": [{"role": "user", "content": prompt}]},
                timeout=60.0,
            )
        except httpx.HTTPError as exc:
            return index, None, time.perf_counter() - started, type(exc).__name__

        elapsed = time.perf_counter() - started
        if response.status_code != 200:
            detail = response.json().get("detail", response.text)
            return index, None, elapsed, f"{response.status_code} {detail}"
        return index, response.headers.get("x-worker-id", "?"), elapsed, None


async def run(args):
    semaphore = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient() as http:
        results = await asyncio.gather(
            *(
                send_one(http, args.url, args.model, args.prompt, semaphore, index)
                for index in range(args.n)
            )
        )

    served = Counter()
    for index, worker_id, elapsed, error in sorted(results):
        if error is None:
            print(f"  {index:>3}  {worker_id:<12} {elapsed * 1000:>7.0f}ms")
            served[worker_id] += 1
        else:
            print(f"  {index:>3}  {'failed':<12} {elapsed * 1000:>7.0f}ms  {error}")
            served["failed"] += 1

    print(f"\n  {args.n} requests, {args.concurrency} at a time")
    for worker_id, count in sorted(served.items()):
        print(f"  {worker_id:<12} {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="mock-model")
    parser.add_argument("--prompt", default="hello")
    parser.add_argument("--n", type=int, default=12, help="how many requests to send")
    parser.add_argument("--concurrency", type=int, default=6, help="how many to keep in flight")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
