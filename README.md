# 5-Stage-Pipelined-RISC-V-Processor


A classic 5-stage in-order RISC-V core (RV32IM) extended with a
tightly-coupled matrix-multiply and attention accelerator, targeting
the core compute kernels of transformer inference (Q·Kᵀ, softmax-
weighted V aggregation, and dense matmul for the FFN layers) as custom
instructions/coprocessor operations rather than pure software loops.


## OverView

  **Base core**: standard 5-stage pipeline — IF → ID → EX → MEM → WB —
      with full hazard handling (forwarding, load-use stall, branch flush)
  **ISA extension**: custom instructions for tile-based matrix
      multiply-accumulate and a fused scaled-dot-product-attention op,
      decoded alongside the base RV32IM instruction set
  **Matmul unit**: systolic-array-style MAC grid (parametrized tile
      size) fed from a local scratchpad, accumulating in fixed-point
      with configurable precision (INT8/INT16 accumulate)
  **Attention unit**: hardware support for Q·Kᵀ score computation,
      softmax normalization (piecewise-linear/LUT-based exp
      approximation), and weighted V aggregation
  **Coupling**: accelerator operates as a tightly-coupled functional
      unit off the EX stage (not a separate bus-attached peripheral),
      minimizing data-movement overhead for small-to-medium tile sizes
  **Software stack**: a minimal C runtime + intrinsics wrapping the
      custom instructions, plus a reference Python/NumPy transformer
      inference implementation for correctness comparison
  **Verification**: core-level ISA compliance, accelerator functional
      tests against the NumPy reference, end-to-end inference of a small
      transformer block compared bit-for-bit (fixed-point) against the
      floating-point reference within a defined error bound
