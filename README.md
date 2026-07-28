# 5-Stage-Pipelined-RISC-V-Processor


A classic 5-stage in-order RISC-V core (RV32IM) extended with a
tightly-coupled matrix-multiply and attention accelerator, targeting
the core compute kernels of transformer inference (Q·Kᵀ, softmax-
weighted V aggregation, and dense matmul for the FFN layers) as custom
instructions/coprocessor operations rather than pure software loops.
