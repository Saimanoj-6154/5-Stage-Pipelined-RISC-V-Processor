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
├── Makefile                          
├── docs/
│   ├── microarchitecture.md          
│   ├── isa_extension.md              
│   ├── accelerator_design.md         
│   └── verification_plan.md          
│
├── rtl/
│   ├── core/
│   │   ├── if_stage.sv
│   │   ├── id_stage.sv
│   │   ├── ex_stage.sv
│   │   ├── mem_stage.sv
│   │   ├── wb_stage.sv
│   │   ├── hazard_unit.sv            
│   │   └── regfile.sv
│   ├── accel/
│   │   ├── matmul_unit.sv            
│   │   ├── mac_pe.sv                 
│   │   ├── scratchpad.sv             
│   │   ├── attention_unit.sv         
│   │   └── softmax_approx.sv         
│   ├── isa/
│   │   ├── custom_decoder.sv         
│   │   └── pkg_custom_isa.sv         
│   ├── common/
│   │   └── pkg_riscv_defs.sv
│   └── top/
│       └── core_top.sv
│
├── sw/
│   ├── runtime/
│   │   ├── intrinsics.h              
│   │   └── crt0.S
│   ├── kernels/
│   │   ├── matmul_test.c
│   │   └── attention_block.c        
│   └── linker/
│       └── core.ld
│
├── verif/
│   ├── ref_model/
│   │   ├── numpy_transformer_ref.py  
│   │   └── fixed_point_model.py      
│   ├── tb/
│   │   ├── core_tb.sv
│   │   ├── matmul_unit_tb.sv
│   │   ├── attention_unit_tb.sv
│   │   └── system_tb.sv              
│   ├── sva/
│   │   ├── hazard_assertions.sv
│   │   └── accel_handshake_assertions.sv
│   └── vectors/
│       └── gen_test_vectors.py       
│
├── sim/
│   └── verilator/
│       ├── Makefile
│       └── sim_main.cpp
│
├── analysis/
│   └── accuracy_report.py            
│
├── scripts/
│   ├── build_sw.sh
│   └── run_regression.py
│
└── .github/
    └── workflows/
        └── ci.yml                    
```

---

## Tools

- Verilator ≥ 5.0
- RISC-V GNU toolchain (`riscv32-unknown-elf-gcc`)
- Python 3.10+ with NumPy (reference model, accuracy analysis)
- GTKWave for waveform debug


## Verification Approach

- **Core ISA compliance**: base RV32IM correctness checked independent
  of the accelerator extension.
- **Accelerator unit tests**: matmul and attention units validated in
  isolation against generated test tensors and a fixed-point golden
  model before integration.
- **Reference model parity**: `verif/ref_model/numpy_transformer_ref.py`
  implements the same small transformer block in floating point;
  `fixed_point_model.py` mirrors the accelerator's fixed-point
  precision. RTL output is checked against both, with error bounds
  tracked explicitly rather than assumed.
- **End-to-end system test**: the core runs `attention_block.c` in
  simulation, exercising the full custom-instruction path from fetch
  through accelerator execution to result writeback.
- **Assertions (SVA)**: pipeline hazard correctness (no missed
  forward/stall case around custom-instruction latency) and accelerator
  handshake protocol (start/done signaling never misaligned with
  pipeline stall behavior).

