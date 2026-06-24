# Agentic GPU Kernel Systems

> A research note repository for studying how LLM agents can generate, test,
> profile, and iteratively optimize GPU kernels.

This repository is inspired by the FlashInfer MLSys 2026 AI Kernel Generation
Contest and my earlier [matmul optimization repo](https://github.com/YupengHan/matmul_optimizer).
In that earlier project, I explored whether quick-feedback kernel optimization
tasks can be automated through agentic loops driven by code generation,
benchmarking, and profiling feedback.

After the contest results and reports were released, this repo uses them as a
case study to review how different teams approached agentic kernel optimization,
summarize their findings, and identify research directions worth pursuing.

## TL;DR

LLMs are no longer just capable of writing CUDA code in isolation. With a strong
execution harness around them - compilation, correctness checks, benchmarking,
profiling, and iterative feedback - they can improve GPU kernels in a closed
loop.

For quick-feedback tasks such as CUDA kernel function generation, the central
question is shifting:

- Not only: can an LLM optimize a kernel?
- But also: how can an agent converge faster, explore better directions, and
  retain reusable optimization knowledge across attempts?

This repo studies the FlashInfer contest as one concrete case study for that
question.

## Motivation

GPU kernel optimization is a useful setting for studying agentic systems because
it has a tight feedback loop:

- Generated code can be compiled immediately.
- Correctness can be checked against reference implementations.
- Runtime can be benchmarked repeatedly.
- Profiling tools can expose bottlenecks.
- Failed attempts still produce useful information for the next attempt.

That makes kernel optimization different from many open-ended coding tasks. The
agent can receive objective signals quickly, but it still needs good strategy:
choosing what to try, interpreting profiler output, avoiding repeated mistakes,
and reusing prior knowledge.

## Why FlashInfer MLSys 2026 Contest?

The FlashInfer contest is a useful public case study because it brings together
multiple systems that attack similar GPU kernel generation problems with
different agent designs.

The representative teams below are not meant to be the full story. Many other
groups also contributed strong technical ideas, engineering patterns, and
analysis worth studying. The goal of this repo is to start from several visible
systems, then broaden the comparison as more reports, code, and writeups are
collected.

## Contest Background

### Track A: Fused MoE

### Track B: DeepSeek Sparse Attention

### Track C: Gated DeltaNet

## Agent-Assisted vs. Full-Agent

## Representative Systems and Techniques

| Team / System | Main idea | Why it matters |
| --- | --- | --- |
| LLM-CUDA | Harness engineering + LoongFlow Trace DB | Treats optimization as an experiment loop with durable traces. |
| UW SyFI | Triton-to-CUDA two-stage loop + `history.json` | Keeps optimization history explicit so the agent can avoid repeating work. |
| Dogacel | Durable artifacts + workload inspector + tiered benchmarking | Separates fast local feedback from more expensive validation. |
| HAN Lab | KernelWiki + `ncu-report-skill` + verifier loop | Moves the agent from pure zero-shot generation toward example-grounded optimization. |

These examples suggest that the core contribution is often not a single prompt.
It is the surrounding system: artifacts, profilers, trace databases, verifier
loops, benchmark tiers, and curated kernel knowledge.

### LLM-CUDA: Harness Engineering + LoongFlow Trace DB

### UW SyFI: Triton-to-CUDA Two-Stage Loop + `history.json`

### Dogacel: Durable Artifacts + Workload Inspector + Tiered Benchmarking

### HAN Lab: KernelWiki + `ncu-report-skill` + Verifier Loop

## My Main Takeaways

1. The harness is part of the intelligence.
2. Profiling feedback must be compressed, not dumped wholesale.
3. Durable memory matters when optimization requires many failed attempts.
4. Few-shot kernel knowledge may be more useful than generic CUDA advice.
5. The remaining problem is experiment strategy, not just code generation.

## Research Directions

### 1. KernelWiki

Build or study a reusable library of high-quality kernel examples that an agent
can retrieve during optimization.

Questions:

- What should a kernel entry contain?
- How should examples be indexed?
- How can the agent tell whether an example is relevant to a new workload?

### 2. Learning to Experiment

Treat kernel optimization as sequential experimental design.

Questions:

- How should an agent choose the next optimization attempt?
- How should it trade off safe incremental changes versus broad exploration?
- Can failed attempts be converted into reusable optimization knowledge?

### 3. Profiling-Aware Agent Memory

Study how profiling summaries, benchmark results, compiler errors, and code
diffs should be stored across attempts.

Questions:

- What memory format helps the model reason most effectively?
- Which signals become stale or misleading?
- Can memory be shared across workloads, devices, or kernel families?

## Related Prior Work

### Karpathy `autoresearch`

Relevant because it frames research itself as an automated loop of hypotheses,
experiments, results, and iteration.

### LoongFlow

Relevant because it emphasizes trace-based workflow construction and reuse,
which maps naturally onto kernel optimization attempts.

## TODO

- [ ] Collect official contest reports and links.
- [ ] Summarize each representative team's system design.
- [ ] Compare what each system persists across iterations.
- [ ] Extract common agent-loop patterns.
- [ ] Identify failure modes that still require human expertise.
- [ ] Turn the research directions into concrete experiments.

## References

- FlashInfer MLSys 2026 AI Kernel Generation Contest
- [YupengHan/matmul_optimizer](https://github.com/YupengHan/matmul_optimizer)
