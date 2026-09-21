"""Shared test helpers."""

import asyncio


async def until(predicate, timeout_s=2.0):
    """Wait for a condition by polling, rather than sleeping a fixed amount.

    A fixed sleep has to be long enough for the slowest machine that will ever
    run it, which makes the suite slow everywhere else, and is still a guess.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition never became true")
