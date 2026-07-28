# 5-Stage-Pipelined-RISC-V-Processor


A classic 5-stage in-order RISC-V core (RV32IM) extended with a
tightly-coupled matrix-multiply and attention accelerator, targeting
the core compute kernels of transformer inference (Q·Kᵀ, softmax-
weighted V aggregation, and dense matmul for the FFN layers) as custom
instructions/coprocessor operations rather than pure software loops.


## OverView

- **Base core**: standard 5-stage pipeline — IF → ID → EX → MEM → WB —
  with full hazard handling (forwarding, load-use stall, branch flush)
- **ISA extension**: custom instructions for tile-based matrix
  multiply-accumulate and a fused scaled-dot-product-attention op,
  decoded alongside the base RV32IM instruction set
- **Matmul unit**: systolic-array-style MAC grid (parametrized tile
  size) fed from a local scratchpad, accumulating in fixed-point
  with configurable precision (INT8/INT16 accumulate)
- **Attention unit**: hardware support for Q·Kᵀ score computation,
  softmax normalization (piecewise-linear/LUT-based exp
  approximation), and weighted V aggregation
- **Coupling**: accelerator operates as a tightly-coupled functional
  unit off the EX stage (not a separate bus-attached peripheral),
  minimizing data-movement overhead for small-to-medium tile sizes
- **Software stack**: a minimal C runtime + intrinsics wrapping the
  custom instructions, plus a reference Python/NumPy transformer
  inference implementation for correctness comparison
- **Verification**: core-level ISA compliance, accelerator functional
  tests against the NumPy reference, end-to-end inference of a small
  transformer block compared bit-for-bit (fixed-point) against the
  floating-point reference within a defined error bound


## Architecture Overview

```
        ┌────┐     ┌────┐     ┌────────────────────┐     ┌─────┐     ┌────┐
   ────▶│ IF │────▶│ ID │────▶│         EX           │────▶│ MEM │────▶│ WB │
        └────┘     └────┘     │  ┌────────────────┐  │     └─────┘     └────┘
                               │  │   Base ALU      │  │
                               │  └────────────────┘  │
                               │  ┌────────────────┐  │
                               │  │  Matmul Unit    │  │  systolic MAC grid,
                               │  │ (tile MAC array)│  │  local scratchpad
                               │  └────────────────┘  │
                               │  ┌────────────────┐  │
                               │  │ Attention Unit  │  │  Q·Kᵀ, softmax approx,
                               │  │ (score+softmax  │  │  weighted V aggregate
                               │  │  +weighted-sum) │  │
                               │  └────────────────┘  │
                               └──────────┬───────────┘
                                          ▼
                             custom instr result forwarded
                             into MEM/WB like any ALU op
```

---

## Repository Structure

```
riscv5-matmul-attention/
├── README.md
├── LICENSE
├── .gitignore
├── Makefile                          # top-level: make sim / make regress / make sw
├── docs/
│   ├── microarchitecture.md          # 5-stage pipeline + hazard handling spec
│   ├── isa_extension.md              # custom instruction encodings, semantics
│   ├── accelerator_design.md         # matmul + attention unit datapath, precision
│   └── verification_plan.md          # test plan, reference-model comparison methodology
│
├── rtl/
│   ├── core/
│   │   ├── if_stage.sv
│   │   ├── id_stage.sv
│   │   ├── ex_stage.sv
│   │   ├── mem_stage.sv
│   │   ├── wb_stage.sv
│   │   ├── hazard_unit.sv            # forwarding, stall, flush logic
│   │   └── regfile.sv
│   ├── accel/
│   │   ├── matmul_unit.sv            # systolic MAC array, parametrized tile size
│   │   ├── mac_pe.sv                 # single processing element
│   │   ├── scratchpad.sv             # local operand/accumulator memory
│   │   ├── attention_unit.sv         # Q.K^T + softmax + weighted-V datapath
│   │   └── softmax_approx.sv         # piecewise-linear/LUT exp approximation
│   ├── isa/
│   │   ├── custom_decoder.sv         # decodes matmul/attention custom opcodes
│   │   └── pkg_custom_isa.sv         # custom instruction encodings
│   ├── common/
│   │   └── pkg_riscv_defs.sv
│   └── top/
│       └── core_top.sv
│
├── sw/
│   ├── runtime/
│   │   ├── intrinsics.h              # C wrappers for custom instructions
│   │   └── crt0.S
│   ├── kernels/
│   │   ├── matmul_test.c
│   │   └── attention_block.c         # small transformer block using the accel ops
│   └── linker/
│       └── core.ld
│
├── verif/
│   ├── ref_model/
│   │   ├── numpy_transformer_ref.py  # floating-point reference: matmul + attention block
│   │   └── fixed_point_model.py      # fixed-point golden model matching accel precision
│   ├── tb/
│   │   ├── core_tb.sv                # ISA-level core sim, RV32IM compliance
│   │   ├── matmul_unit_tb.sv
│   │   ├── attention_unit_tb.sv
│   │   └── system_tb.sv              # end-to-end: core running attention_block.c
│   ├── sva/
│   │   ├── hazard_assertions.sv      # no incorrect forward/stall combination
│   │   └── accel_handshake_assertions.sv
│   └── vectors/
│       └── gen_test_vectors.py       # generates matmul/attention test tensors
│
├── sim/
│   └── verilator/
│       ├── Makefile
│       └── sim_main.cpp
│
├── analysis/
│   └── accuracy_report.py            # fixed-point vs. floating-point error analysis
│
├── scripts/
│   ├── build_sw.sh
│   └── run_regression.py
│
└── .github/
    └── workflows/
        └── ci.yml                    # lint + core-level smoke tests on push
```

---

