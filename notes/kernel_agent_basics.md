# Kernel Agent Basics

这些是阅读 KDA / Kernel Mafia team 的 FlashInfer Full-Agent workflow 时用到的基础概念。它们更像个人学习词条，不属于主 report 的论证主体。

## TorchInductor

TorchInductor 是 PyTorch `torch.compile` 的默认 compiler backend。直觉上，TorchDynamo 先把 Python/PyTorch 程序 capture 成 graph，TorchInductor 再把 graph lowering 成更底层的高性能代码；在 GPU 上通常生成 Triton code，在 CPU 上可以生成 C++/OpenMP 风格代码。它的目标不是让用户手写 kernel，而是自动做 graph capture、fusion、layout/shape specialization、kernel generation 和 runtime wrapper 管理。PyTorch 文档也说明 TorchInductor 是默认 `torch.compile` deep learning compiler，并且在 NVIDIA/AMD/Intel GPU 上以 Triton 作为关键 building block。

## Schedule 设计

Schedule 设计不是简单“backend 有很多 kernel，然后 tree search 选一个”。更准确地说，schedule 是把一个数学计算映射到硬件执行的方案。对于 GEMM / attention 这类算子，schedule 包括 tile size、block/warp/thread 分工、loop order、shared memory staging、register blocking、vectorized load、software pipeline、warp specialization、TMA/cp.async 使用、epilogue fusion 等。

TensorRT-LLM / cuDNN / CUTLASS 里按 shape 选择 tactic 或 kernel 更偏 **kernel selection / dispatch**；TVM / Triton / TorchInductor 里的 schedule 更偏 **如何生成这个 kernel 本身**。KDA technical report 说 TVM、Triton、TorchInductor 能自动化一部分 kernel development，但强性能仍然依赖 expert-written templates 和 architecture-aware schedule design。

## KernelBench

KernelBench 不是“已经高度优化的 kernel 代码库”，而是一个 benchmark / evaluation environment，用来评测 LLM 是否能为 PyTorch workloads 生成正确且更快的 CUDA / DSL kernel。它包含多层任务：single-kernel operators、simple fusion patterns、full model architectures 等，并通过 correctness check 与 speedup 计算 `fast_p` 指标。KernelBench repo 也明确说它的任务是让 LLM 为 PyTorch programs 生成 correct and efficient CUDA / DSL kernels，而不是提供一组最优 kernel 给用户直接用。

为什么说 KernelBench 容易把 LLM 当代码生成器？因为最基础的 KernelBench workflow 是：给模型一个 PyTorch reference，让模型输出 CUDA/Triton kernel，然后 evaluation harness 检查 correctness 和 latency。即使后面加入 execution feedback 或 profiling feedback，模型的主要角色仍然常常是 candidate generator；外层 pipeline 决定何时生成、何时评测、如何汇总结果。KDA technical report 批评的不是 KernelBench 本身，而是这种固定 pipeline 容易遮蔽真实 kernel engineering 中更复杂的 loop：读 repo、修 harness、profile、诊断瓶颈、查知识库、回退、长期迭代。
