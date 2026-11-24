# Cross-project research: Neuro-Silica, Sogitech, and Metronome

**Author:** Lewis James Seals Jr. (ljrseals-dev)  
**SPDX-License-Identifier:** MIT  
**Status:** RESEARCH PLAN — NOT EXECUTED

## Scope and ownership

This document defines experiments between two independently maintained
repositories:

- [ljrseals-dev/CPU](https://github.com/ljrseals-dev/CPU): limited AArch64
  execution and fixed-point arithmetic research.
- [ljrseals-dev/Sogitech](https://github.com/ljrseals-dev/Sogitech): hybrid
  DSP, parameter codec, and experimental neural parameter controller.

[brentius/metronome](https://github.com/brentius/metronome) is an external,
read-only research reference. Its owner controls its source and ongoing
development. No changes, copied code, or assumed test results are made there.

## Metronome method inspected

Metronome v4's code/tests/machine_lockstep/README.md documents a per-cycle
lockstep harness between a Lean executable spec and a Python model. It compares
outputs plus a 197-key state dump, supports directed and random scenarios,
self-tests mismatch detection with a fake simulator that flips a selected bit,
and reports the earliest divergence for disciplined triage.

The Sogitech adaptation is property-oriented rather than cycle-oriented:
compare parameter encode/decode invariants, quantization boundaries, and
deterministic renderer outputs; inject deliberate one-bit/one-field faults
into test doubles to prove that checks detect them. The Metronome harness is
not imported and its ASIC/RTL-specific implementation is not a codec backend.

## Hypotheses worth testing

1. **Fixed-point conformance:** use Sogitech quantization/dequantization vectors
   as deterministic arithmetic fixtures for Neuro-Silica's MEMT1 fallback and
   future guest arithmetic. Compare exact integer outputs, overflow behavior,
   and boundary values.
2. **Quantizer invariants:** use CPU's state-trace discipline to define
   Sogitech invariants for encode/decode round trips, monotonicity, saturation,
   and deterministic output.
3. **Performance portability:** compare scalar reference operations with
   future optimized implementations only after the reference result is
   independently established. Do not assume that the current CPU interpreter
   accelerates Python or Cmajor workloads.
4. **Verification transfer:** inspect Metronome's exact sizing and verification
   methods before adapting any concept. Latch-bank sizing or physical-design
   flows are not automatically applicable to DSP or an AArch64 interpreter.

## Candidate search policy

The default Sogitech space declares 11 dimensions and 23,328 unconstrained
combinations. Local constrained enumeration now applies the declared
conditional rule and yields an expected 15,552 feasible configurations.

The enumerator is a planner primitive, not a codec evaluator. It does not
produce audio, train a controller, launch remote workers, or generate measured
quality/latency scores. Each future evaluator must use fixed source-grouped
fixtures, a fixed seed per candidate, repeated measurements, and the
Cmajor implementation as the real-time authority.

## Required experiment record

For every candidate, store:

- candidate ID and canonical parameter JSON;
- repository and commit SHAs;
- fixture hashes and train/validation/test source groups;
- toolchain, host architecture, sample rate, and seed;
- correctness status and reference comparison;
- spectral L1, multi-resolution STFT, parameter L1, residual-energy error,
  peak dBFS, measured latency, and CPU load where applicable;
- repetitions, raw measurements, aggregate statistics, and failure reason.

Reject correctness failures before ranking. Never substitute estimated latency
for measured latency. Never use a missing neural checkpoint or random weights
as an inference candidate.

## Integration gates

- CPU MADD/MSUB regression and full native/Node/WASM suite pass.
- Sogitech candidate enumerator and planner tests pass.
- Differential fixed-point fixtures agree with an independent reference.
- Audio candidate evaluations pass offline checks and Cmajor compile/host
  validation before real-time claims.
- Metronome techniques are cited to exact source files/commits and reviewed
  for license and domain compatibility before any adaptation.

## Current status

Authored on the research branch. No candidate audio evaluations, neural
training runs, CPU benchmark runs, Cmajor compilation, or Metronome jobs are
claimed as completed.
