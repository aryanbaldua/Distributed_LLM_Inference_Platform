# Distributed LLM Inference Platform

## V0 High-Level Design

## 1. Overview

The project will provide a single inference endpoint backed by multiple worker processes or machines. Each worker runs the same supported LLM through an existing inference engine such as vLLM. A controller maintains a live view of the cluster and decides which healthy worker should receive each request.

The goal of V0 is not to optimize the model itself, but to build and understand the distributed infrastructure required to coordinate independent inference workers.

> **V0 in one sentence:** Build a small cluster in which multiple LLM inference workers appear to clients as one service, with centralized request routing, worker discovery, basic load-aware scheduling, health monitoring, and failure handling.

### 1.1 Goals

- Expose one client-facing inference service backed by multiple workers.
- Allow workers to register, send heartbeats, join the cluster, and disappear from the cluster.
- Route requests using a simple load-aware scheduling policy.
- Detect unavailable workers and stop routing new requests to them.
- Support streamed model responses through the controller.
- Collect enough metrics and logs to observe routing, worker health, load, and failures.
- Run the system with at least two independent workers and demonstrate its behavior under load and failure.

### 1.2 Non-goals for V0

- Training, fine-tuning, or modifying an LLM.
- Distributed KV-cache sharing, prefix-aware routing, or other advanced inference optimizations.
- Prefill/decode disaggregation, tensor parallelism across machines, or custom CUDA kernels.
- Automatic model placement, arbitrary multi-model serving, or model migration between workers.
- Production-grade authentication, billing, quotas, multi-tenancy, or user management.
- Kubernetes-based orchestration or sophisticated cloud autoscaling.
- A highly available controller or consensus protocol. The V0 controller may be a single point of failure.

## 2. System Architecture

V0 uses a simple control-plane/data-plane split. The controller owns cluster membership and scheduling decisions. Workers own model execution. Clients interact only with the controller, so the cluster appears as one inference service.

```text
                  Client / Application
                           |
                           v
                  Controller / Router
                           |
                  scheduling + forwarding
                    /      |      \
                   v       v       v
               Worker A Worker B Worker C
               vLLM+GPU vLLM+GPU vLLM+GPU
                   \       |       /
                    heartbeats + metrics
```

### 2.1 Main Components

| Component | Responsibility | Key V0 State | Likely Tech |
| --- | --- | --- | --- |
| Client/API | Submit chat/completion requests and receive streamed output. | Request payload and stream. | HTTP |
| Controller | Track workers, schedule requests, forward traffic, and expose cluster state. | Worker registry, health, active load. | Python + FastAPI |
| Worker | Run one supported LLM and report health/load to the controller. | Model, queue/load, GPU stats. | Python + vLLM |
| Metrics | Capture request and cluster behavior for debugging and evaluation. | Latency, throughput, errors, worker health. | Logs + Prometheus |

> **Important V0 assumption:** The controller is authoritative for cluster membership. V0 does not attempt decentralized membership, consensus, or controller replication.

## 3. Core V0 Flows

### 3.1 Worker Registration and Membership

When a worker starts, it registers with the controller and announces the model it serves plus basic capacity information. The controller stores the worker in an in-memory registry. The worker then sends periodic heartbeats containing current load and health information.

If heartbeats stop for longer than a configured timeout, the controller marks the worker unhealthy and removes it from scheduling eligibility.

```text
1. Worker starts and initializes the model server.
2. Worker sends REGISTER(worker_id, address, model, capacity).
3. Controller adds the worker to the active registry.
4. Worker periodically sends HEARTBEAT(load, health, GPU metrics).
5. Controller updates the worker record and last-seen timestamp.
6. If the timeout is exceeded, the controller marks the worker unavailable.
```

### 3.2 Request Routing

The controller receives an inference request, filters the registry to workers that are healthy and serving the required model, then applies a simple scheduling policy. The initial policy should prefer the worker with the lowest active-request count or shortest known queue.

The controller forwards the request and streams generated tokens back to the client.

```text
Client
  |
  v
Controller
  |
  +--> identify eligible healthy workers
  +--> choose worker using current load
  +--> forward request
            |
            v
          Worker
            |
            v
          vLLM
            |
            v
      generated tokens
            |
            v
Controller --> Client
```

Request lifecycle:

1. Client sends an inference request to the controller.
2. Controller identifies eligible healthy workers.
3. Scheduler ranks workers using current load.
4. Controller forwards the request to the selected worker.
5. Worker performs inference through vLLM.
6. Generated tokens stream from worker to controller to client.
7. Controller updates metrics when the request finishes or fails.

### 3.3 Worker Failure

Failure handling is intentionally basic.

- If a worker becomes unreachable before a request is assigned, it is skipped.
- If forwarding fails before generation has meaningfully started, the controller may retry the request once on another healthy worker.
- If a worker fails after streamed output has already been delivered, V0 may terminate the stream and return an error rather than attempting transparent continuation.

This keeps V0 focused on detecting failures and routing around them without solving distributed checkpointing or generation recovery.

## 4. Scheduling and Cluster State

### 4.1 Worker State

The controller maintains a small record for each worker. The exact fields can evolve, but V0 should include enough information to answer three questions:

1. Is this worker alive?
2. Can it serve this request?
3. How busy is it?

| Field | Example | Purpose |
| --- | --- | --- |
| `worker_id` | `gpu-worker-02` | Stable identity for the worker. |
| `address` | `10.0.0.22:8001` | Where inference requests are sent. |
| `model` | `Qwen-family model` | Confirms request compatibility. |
| `status` | `HEALTHY` / `UNHEALTHY` | Controls scheduling eligibility. |
| `last_heartbeat` | timestamp | Supports failure detection. |
| `active_requests` | `3` | Primary V0 load signal. |
| `gpu_utilization` | `72%` | Useful metric; optional scheduler input. |
| `free_vram` | `8 GB` | Useful metric; optional scheduler input. |

### 4.2 Initial Scheduling Policy

The first scheduler should be deliberately understandable rather than sophisticated.

1. Filter to healthy workers serving the required model.
2. Choose the worker with the fewest active requests.
3. Break ties randomly or round-robin.

Conceptually:

```python
eligible = [
    worker for worker in workers
    if worker.status == "HEALTHY" and worker.model == requested_model
]

selected = min(eligible, key=lambda worker: worker.active_requests)
```

This policy becomes the baseline for later GPU-aware and cache-aware schedulers.

## 5. External and Internal Interfaces

The exact API schema can evolve during implementation. V0 only needs a small set of interfaces that make the distributed behavior explicit.

| Interface | Direction | Purpose |
| --- | --- | --- |
| `POST /v1/chat/completions` | Client -> Controller | Submit a request and receive a streamed response. |
| `POST /workers/register` | Worker -> Controller | Join the cluster and advertise capabilities. |
| `POST /workers/heartbeat` | Worker -> Controller | Refresh health and load information. |
| `GET /cluster/workers` | Operator -> Controller | Inspect current cluster membership and state. |
| Inference endpoint | Controller -> Worker | Forward a selected request to the model server. |

The public inference interface should eventually be OpenAI-compatible where practical so existing clients can call the service without knowing how the cluster is implemented.

## 6. Observability and V0 Validation

V0 should be judged by observable distributed behavior, not just whether the model returns text. Logs and metrics should make scheduling and failures visible enough to explain what the cluster is doing.

### 6.1 Minimum Metrics

- Requests received, completed, and failed.
- Per-worker active request count and health status.
- Request latency and time to first token where practical.
- Which worker handled each request.
- Heartbeat age and worker failure events.
- Optional GPU utilization and VRAM usage.

### 6.2 V0 Demo Scenarios

#### Normal operation

Send multiple requests and show that traffic is distributed across at least two workers.

#### Load imbalance

Artificially load one worker and show that new requests prefer a less-busy worker.

#### Worker failure

Stop one worker and show that the controller marks it unhealthy and stops routing to it.

#### Worker recovery

Restart the worker and show that it registers or becomes healthy again and reenters scheduling.

#### Streaming

Show a client receiving tokens through the controller rather than connecting directly to a worker.

## 7. Initial Implementation Approach

Development should begin in the simplest environment that preserves the architecture. The controller and multiple worker processes can initially run on one machine or a small number of machines. Once the control flow is stable, workers can be moved to separate GPU hosts without changing the logical design.

| Area | V0 Choice |
| --- | --- |
| Language | Python for controller and worker services. |
| Public API | FastAPI / HTTP with streaming responses. |
| Inference engine | vLLM running one supported open-source model. |
| Packaging | Docker after the first local prototype works. |
| Cluster state | In-memory state in the controller for V0. |
| Metrics | Structured logs first; Prometheus integration if time permits. |
| Deployment | Local multi-process or a few manually provisioned machines. Kubernetes is out of scope. |

## 8. V0 Design Risks and Simplifications

| Risk / Simplification | V0 Treatment |
| --- | --- |
| Controller failure | The controller is a single point of failure. Accepted for V0; controller replication is deferred. |
| Stale load information | Heartbeats provide an approximate view of worker load. Scheduling may occasionally use stale state; this is acceptable for the baseline. |
| Mid-stream worker failure | Transparent continuation is difficult once tokens have been emitted. V0 returns an error instead of reconstructing the stream elsewhere. |
| GPU availability | Development may not always have several physical GPUs. Multiple workers can be simulated where needed, then validated on a smaller real GPU cluster. |
| Scope growth | Advanced inference optimizations are intentionally deferred so V0 remains focused on core distributed coordination. |

## 9. Later Versions

Later versions build on the same V0 architecture. The intent is to add capability incrementally rather than redesign the entire system at once.

| Version | Theme | What It Adds |
| --- | --- | --- |
| V1 | Smarter scheduling | More complete observability plus GPU-aware scheduling using queue depth, utilization, VRAM pressure, request characteristics, and benchmark comparisons against the V0 baseline. |
| V2 | Elastic cluster management | Autoscaling, improved model placement, more robust cluster state, and potentially multi-model deployments. |
| V3 | AI-specific optimization | Prefix/cache-aware routing and distributed KV-cache techniques for reusing LLM computation across workers. |
| V4+ | Advanced inference architecture | Potential prefill/decode disaggregation, higher-performance data movement, and stronger control-plane fault tolerance if the project evolves that far. |

## 10. V0 Success Criteria

V0 is complete when the system can demonstrate the following end-to-end behavior:

- At least two independent workers can register with one controller and serve the same model.
- Clients use one controller endpoint and do not need to know worker addresses.
- The controller makes visible load-aware scheduling decisions.
- Workers are removed from scheduling after missing heartbeats and can rejoin after recovery.
- Inference output can stream through the controller to the client.
- Logs or metrics clearly show worker membership, routing decisions, load, and failure events.
- A repeatable demo or test harness proves normal routing, load imbalance handling, failure detection, and recovery.

> **Design principle:** Keep V0 simple enough that every major distributed-systems mechanism can be explained and debugged directly. Industrial orchestration and AI-specific optimizations should be added only after the baseline behavior is understood and measurable.
