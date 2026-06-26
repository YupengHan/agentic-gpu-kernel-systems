# Agentic GPU Kernel Systems

> A research note repository studying **agentic + harness loops** as a primitive for **fast-and-verifiable tasks**, with **CUDA kernel generation and optimization** as a high-value case study. This repo distills lessons from the top teams in the FlashInfer AI Kernel Generation Contest, which I discovered too late to join but now want to study deeply after the contest has concluded.

This repository is inspired by the [FlashInfer MLSys 2026 AI Kernel Generation](https://mlsys26.flashinfer.ai/)[Contest](https://mlsys26.flashinfer.ai/) and my earlier [matmul optimization repo](https://github.com/YupengHan/matmul_optimizer).

My matmul project was an initial attempt to structure CUDA kernel optimization around generation, verification, benchmarking, and profiling. The FlashInfer contest revealed how far this approach can scale when combined with stronger evaluation pipelines, richer experiment tracking, and high-quality kernel references.

Because I discovered the contest too late to participate, this repository is my post-contest study notebook. It reviews the top teams' reports, extracts common system patterns, and organizes the open research questions that seem worth pursuing next.

## Public Reading Pages

- [Repository README](https://github.com/YupengHan/agentic-gpu-kernel-systems#readme)
- [KDA Kernel Mafia FlashInfer Workflow Analysis](https://yupenghan.github.io/agentic-gpu-kernel-systems/flashinfer_contest_han_lab_kernel_mafia_analysis.html)

## TL;DR

LLM-based kernel optimization is moving from isolated code generation toward closed-loop experimental systems. Once compilation, correctness testing, benchmarking, profiling, and regression checks are built into the loop, kernel improvement becomes a measurable search problem rather than a one-shot coding task.

The more interesting question is no longer simply whether an LLM can produce a faster kernel. It is how the surrounding system should guide exploration, reject bad directions, reuse prior experiments, and turn scattered optimization attempts into durable knowledge.

This repo studies the FlashInfer contest through that system's lens.

## Core Insight

GPU kernel optimization is a useful setting for studying agentic + harness loops because it turns an otherwise open-ended coding problem into a fast-feedback, low-noise experimental loop:

- Feedback is end-to-end: the output is the final kernel behavior, not an indirect internal proxy.
- The success signal is explicit: compile, correctness, latency, and profiler evidence are directly observable.
- The measurement target is narrow enough to compare attempts without heavy judgment.
- The evaluation is cheap enough to run many times and stable enough to trust across repetitions.
- Negative results are still informative because they constrain the search space.

This structure gives LLM agents a much more stable operating regime. Each update can stay small and local, so the system can move through a sequence of bounded code changes rather than relying on a single large generation. Bad changes can be reverted. Different attempts can be isolated into separate sessions, which makes context easier to manage. Most importantly, the probabilistic behavior of the model is anchored by a deterministic evaluation harness, instead of being allowed to drift without reliable external correction.

<!-- ## Contest Background

### Track A: Fused MoE

### Track B: DeepSeek Sparse Attention

### Track C: Gated DeltaNet

## Agent-Assisted vs. Full-Agent

## Representative Systems and Techniques

| Team / System | Main idea                                                    | Why it matters                                                                       |
| ------------- | ------------------------------------------------------------ | ------------------------------------------------------------------------------------ |
| LLM-CUDA      | Harness engineering + LoongFlow Trace DB                     | Treats optimization as an experiment loop with durable traces.                       |
| UW SyFI       | Triton-to-CUDA two-stage loop + `history.json`               | Keeps optimization history explicit so the agent can avoid repeating work.           |
| Dogacel       | Durable artifacts + workload inspector + tiered benchmarking | Separates fast local feedback from more expensive validation.                        |
| HAN Lab       | KernelWiki + `ncu-report-skill` + verifier loop              | Moves the agent from pure zero-shot generation toward example-grounded optimization. |

These examples suggest that the core contribution is often not a single prompt.
It is the surrounding system: artifacts, profilers, trace databases, verifier
loops, benchmark tiers, and curated kernel knowledge.

### LLM-CUDA: Harness Engineering + LoongFlow Trace DB

### UW SyFI: Triton-to-CUDA Two-Stage Loop + `history.json`

### Dogacel: Durable Artifacts + Workload Inspector + Tiered Benchmarking

### HAN Lab: KernelWiki + `ncu-report-skill` + Verifier Loop -->

## My Main Takeaways

1. For current LLMs, CUDA code generation is already useful, but a strong harness remains a clear capability amplifier.
2. Profiling is essential, but raw reports must become compact, actionable feedback.
3. Context length is finite, so long optimization runs need disk-backed, retrievable experiment memory.
4. Few-shot kernel examples may be more valuable than generic CUDA optimization advice.
5. The open problem is experiment strategy: learning what to try next from accumulated failures and wins.

## Research Directions

### 1. KernelWiki

<!-- Build or study a reusable library of high-quality kernel examples that an agent
can retrieve during optimization.

Questions:

- What should a kernel entry contain?
- How should examples be indexed?
- How can the agent tell whether an example is relevant to a new workload? -->

### 2. Learning to Experiment

<!-- Treat kernel optimization as sequential experimental design.

Questions:

- How should an agent choose the next optimization attempt?
- How should it trade off safe incremental changes versus broad exploration?
- Can failed attempts be converted into reusable optimization knowledge? -->

### 3. Profiling-Aware Agent Memory

<!-- Study how profiling summaries, benchmark results, compiler errors, and code
diffs should be stored across attempts.

Questions:

- What memory format helps the model reason most effectively?
- Which signals become stale or misleading?
- Can memory be shared across workloads, devices, or kernel families? -->

## Related Prior Work

### Karpathy `autoresearch`

<!-- Relevant because it frames research itself as an automated loop of hypotheses, experiments, results, and iteration. -->

### LoongFlow

<!-- Relevant because it emphasizes trace-based workflow construction and reuse, which maps naturally onto kernel optimization attempts. -->

### K-Search

<!-- [K-Search](https://arxiv.org/abs/2602.19128) is directly relevant because it
treats LLM-driven kernel generation as evidence-guided search over bottleneck
hypotheses, design alternatives, and optimization strategies. Its co-evolving
world model is a useful comparison point for the KernelWiki and experiment-memory
directions in this repo. -->

### AVO

<!-- [AVO](https://arxiv.org/abs/2603.24517) is relevant because it elevates the
LLM-backed coding agent from candidate generator to evolutionary variation
operator. The method uses lineage, domain knowledge, and execution feedback to
propose, repair, critique, and verify implementation edits, making it a strong
example of autonomous kernel-search loops. -->

## TODO

- [ ] Study the MIT/HAN Lab report first, then start a dedicated KernelWiki notes file.
- [ ] Summarize each representative team's system design.
- [ ] Compare what each system persists across iterations.
- [ ] Extract common agent-loop patterns.
- [ ] Identify failure modes that still require human expertise.
- [ ] Turn the research directions into concrete experiments.

## References

- [FlashInfer MLSys 2026 AI Kernel Generation Contest](https://mlsys26.flashinfer.ai/)
- [K-Search: LLM Kernel Generation via Co-Evolving Intrinsic World Model](https://arxiv.org/abs/2602.19128)
- [AVO: Agentic Variation Operators for Autonomous Evolutionary Search](https://arxiv.org/abs/2603.24517)
- [YupengHan/matmul_optimizer](https://github.com/YupengHan/matmul_optimizer)
