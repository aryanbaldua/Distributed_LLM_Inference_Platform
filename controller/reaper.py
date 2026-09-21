"""The background task that turns silence into an UNHEALTHY worker.

Detection is pull-based on a timer rather than event-driven, because the event
being detected is the *absence* of a message. Nothing arrives to react to, so
something has to go looking on a schedule.
"""

from __future__ import annotations

import asyncio

from common.config import ControllerSettings
from common.logging import get_logger
from controller.registry import Transition, WorkerRegistry

log = get_logger("controller.reaper")


def sweep_once(
    registry: WorkerRegistry, settings: ControllerSettings, now: float | None = None
) -> list[Transition]:
    """One pass: mark the silent dead, then forget the long dead.

    Split out of the loop so the work can be tested without a running event loop
    or a real clock.
    """
    transitions = registry.sweep(settings.heartbeat_timeout_s, now=now)
    registry.evict(settings.evict_after_s, now=now)
    return transitions


async def reaper_loop(registry: WorkerRegistry, settings: ControllerSettings) -> None:
    """Sweep forever, until cancelled at shutdown.

    Sleeps first, so nothing is judged stale before it has had a chance to
    heartbeat.

    A worker can stay HEALTHY in the registry for up to one tick past its
    timeout, because status is owned by this loop rather than computed on read.
    That is the deliberate trade: owning status here is what produces the
    transition *events* the failure demo is built on. The scheduler should
    re-check heartbeat age at selection time rather than trust the flag to be
    fresh to the millisecond.
    """
    log.info(
        "reaper started (every %.1fs, timeout %.1fs, evict after %.0fs)",
        settings.reaper_interval_s,
        settings.heartbeat_timeout_s,
        settings.evict_after_s,
    )
    while True:
        await asyncio.sleep(settings.reaper_interval_s)
        try:
            sweep_once(registry, settings)
        except Exception:
            # One bad tick must not kill the task. An exception escaping here
            # would leave the cluster with no failure detection at all, and
            # because nothing ever awaits this task, it would do so silently.
            # CancelledError is a BaseException and passes straight through,
            # which is what lets shutdown stop the loop.
            log.exception("reaper tick failed; continuing")
