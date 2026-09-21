"""The worker's half of the membership protocol.

Registration and heartbeating are one task rather than two, because they are
one sequence: a worker cannot heartbeat before it has registered, and a worker
the controller has forgotten has to register again before its heartbeats mean
anything. Splitting them would mean inventing a way for the two to wait on each
other.
"""

from __future__ import annotations

import asyncio
import random
from enum import Enum

import httpx

from common.config import WorkerSettings
from common.logging import get_logger
from common.schemas import HeartbeatRequest, RegisterRequest, RegisterResponse


class HeartbeatOutcome(str, Enum):
    ACCEPTED = "accepted"
    UNKNOWN_WORKER = "unknown_worker"
    UNREACHABLE = "unreachable"


class ControllerClient:
    def __init__(
        self,
        settings: WorkerSettings,
        http: httpx.AsyncClient,
        log=None,
    ) -> None:
        self._settings = settings
        self._http = http
        self._log = log or get_logger(f"worker:{settings.worker_id}")
        self._policy: RegisterResponse | None = None
        # Tracks whether the current outage has already been reported, so a
        # controller that is down for ten minutes produces one line instead of
        # three hundred identical ones burying everything else in the log.
        self._outage_reported = False

    @property
    def _base_url(self) -> str:
        return self._settings.controller_url.rstrip("/")

    # --- registration ---------------------------------------------------------

    async def register(self) -> RegisterResponse:
        """A single attempt. Raises httpx errors for the caller to back off on."""
        request = RegisterRequest(
            worker_id=self._settings.worker_id,
            address=self._settings.advertised_address,
            model=self._settings.model,
            max_concurrency=self._settings.max_concurrency,
        )
        response = await self._http.post(
            f"{self._base_url}/workers/register", json=request.model_dump(mode="json")
        )
        response.raise_for_status()
        return RegisterResponse.model_validate(response.json())

    def backoff_delay(self, attempt: int) -> float:
        """Exponential, capped, and jittered.

        Jittered because workers are usually started together - `run_local.sh`
        launches both at once - and without it they would retry in lockstep for
        as long as the controller stayed down, arriving in the same instant
        every time.
        """
        ceiling = min(
            self._settings.register_retry_s * 2 ** (attempt - 1),
            self._settings.register_backoff_max_s,
        )
        return ceiling * random.uniform(0.5, 1.0)

    async def register_until_accepted(self) -> RegisterResponse:
        """Retry forever. A worker that gave up would need a human to restart it
        for no reason other than being started before the controller."""
        attempt = 0
        while True:
            attempt += 1
            try:
                policy = await self.register()
            except (httpx.HTTPError, ValueError) as exc:
                delay = self.backoff_delay(attempt)
                self._log.info(
                    "controller at %s not accepting registrations (%s); retrying in %.1fs",
                    self._base_url,
                    type(exc).__name__,
                    delay,
                )
                await asyncio.sleep(delay)
                continue

            self._policy = policy
            self._outage_reported = False
            self._log.info(
                "registered with controller as %s; heartbeating every %.1fs (timeout %.1fs)",
                self._settings.worker_id,
                policy.heartbeat_interval_s,
                policy.heartbeat_timeout_s,
            )
            return policy

    # --- heartbeats -----------------------------------------------------------

    async def send_heartbeat(self) -> HeartbeatOutcome:
        """One beat.

        Bounded by the heartbeat interval itself, so a stalled request can never
        eat the slot belonging to the next beat and make a healthy worker look
        silent.
        """
        request = HeartbeatRequest(
            worker_id=self._settings.worker_id,
            # The worker's own in-flight count lands when it starts serving
            # requests. Until then there is nothing to count.
            active_requests=0,
        )
        timeout = self._policy.heartbeat_interval_s if self._policy else None

        try:
            response = await self._http.post(
                f"{self._base_url}/workers/heartbeat",
                json=request.model_dump(mode="json"),
                timeout=timeout,
            )
        except httpx.HTTPError as exc:
            self._report_outage(type(exc).__name__)
            return HeartbeatOutcome.UNREACHABLE

        if response.status_code == httpx.codes.NOT_FOUND:
            return HeartbeatOutcome.UNKNOWN_WORKER
        if response.is_error:
            self._report_outage(f"HTTP {response.status_code}")
            return HeartbeatOutcome.UNREACHABLE

        self._report_recovery()
        return HeartbeatOutcome.ACCEPTED

    def _report_outage(self, detail: str) -> None:
        if self._outage_reported:
            return
        self._outage_reported = True
        self._log.warning(
            "heartbeats to %s are failing (%s); continuing to try", self._base_url, detail
        )

    def _report_recovery(self) -> None:
        if not self._outage_reported:
            return
        self._outage_reported = False
        self._log.info("heartbeats to %s are getting through again", self._base_url)

    # --- the loop -------------------------------------------------------------

    async def run(self) -> None:
        """Register, then heartbeat until cancelled at shutdown.

        A failed heartbeat is not backed off. The next beat goes out on schedule,
        because backing off here would stretch exactly the window in which the
        controller is deciding whether this worker is still alive.
        """
        policy = await self.register_until_accepted()

        while True:
            await asyncio.sleep(policy.heartbeat_interval_s)
            outcome = await self.send_heartbeat()

            if outcome is HeartbeatOutcome.UNKNOWN_WORKER:
                # This process outlived its registry entry - evicted after being
                # unreachable long enough to be written off. Only registration
                # can put it back: a heartbeat carries no address or model for
                # the controller to rebuild the record from.
                self._log.info("controller has forgotten this worker; registering again")
                policy = await self.register_until_accepted()
