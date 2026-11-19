# Engineering Review: `gate-fix-no-cmajor`

**Disposition:** frozen reference branch; not production-ready and not a merge candidate.

- Repository: `ljrseals-dev/Sogitech`
- Reviewed branch: `gate-fix-no-cmajor`
- Reviewed commit: `6d8840e1653806f2bab46605389a9d5fc2b99ccc`
- Baseline: `main` at `1a40ed625743d16e41341a0a3c08d0a2b6f45ff5`
- Branch relationship at review: one commit ahead, zero behind.
- Earlier correctness PR #3 is merged into the baseline. Its merge commit is `1a40ed625743d16e41341a0a3c08d0a2b6f45ff5`.

This note records engineering lessons only. It does not authorize modifying the frozen branch, replacing the Cmajor engine, changing the vintage instrument topology, or merging code from the frozen branch.

## Safe ideas worth retaining

1. **A standalone offline validation path.** A Python reference renderer can make numerical and regression checks accessible without requiring a Cmajor compiler to be installed. It must remain a reference/diagnostic path, not silently become the authoritative instrument implementation.
2. **A reproducible validation harness.** Keep fixtures, parameter inputs, expected invariants, and machine-readable outcomes together. A validation command should fail clearly when prerequisites are missing and should distinguish skipped checks from passed checks.
3. **Separate validation layers.** Report Python reference tests, Cmajor compilation/tests, DAW-host playback, and physical MIDI/audio wiring as separate evidence. Passing one layer must never be described as passing the others.
4. **Document physical/hardware checks.** The frozen branch's hardware-validation documentation is a useful prompt to retain a checklist for real host, interface, MIDI, levels, and signal-path verification. Adapt its concepts; do not copy it blindly.

## Do not carry forward from this branch

The review identified regressions or unresolved correctness risks in the branch's implementation:

- The existing `tests/` suite and `scripts/train_controller.py` were removed.
- The replacement offline renderer did not preserve the documented harmonic/waveform behavior and had a PowerSag attenuation path without recovery.
- New Cmajor processors used fixed 96 kHz assumptions rather than the host's actual sample rate.
- MIDI event/state behavior was not preserved in the replacement patches.
- Neural-controller checkpoint validation and output-range enforcement were weakened; missing checkpoints could yield randomly initialized behavior.
- Analysis and renderer input validation was reduced, and broad exception fallbacks could conceal errors.
- Packaging entry points and documented paths appeared inconsistent with the branch's package layout.
- The branch also removed or weakened documentation of the vintage home-studio topology and monophonic voice policy.

These are code-review findings, not claims of successful runtime reproduction. The frozen branch had no recorded workflow run for its commit, and no Cmajor compiler, DAW host, or physical hardware validation was performed as part of this review.

## Preservation requirements

- Keep the original monophonic voice policy and vintage home-studio signal topology intact.
- Preserve the canonical Cmajor instrument and MIDI behavior; any offline Python renderer is supplementary.
- Treat the historical OpticLink/reference generator as reference material, not a verified hardware schematic or bit-exact emulation.
- Any modern polyphonic, multichannel, or alternate-host mode must be an explicitly separate extension, not a silent redesign.
- Keep the frozen branch available for forensic comparison. Do not cherry-pick its broad rewrite or merge it.

## Recommended validation contract for future work

| Layer | What it can establish | What it cannot establish |
|---|---|---|
| Python reference | Deterministic offline numerical behavior and regression invariants | Cmajor compilation, DAW integration, or physical sound |
| Cmajor toolchain | Patch syntax, compilation, and declared Cmajor tests | Host-specific behavior or physical wiring |
| DAW/host | Plugin/patch loading, host sample-rate behavior, MIDI interaction, playback | Correctness of every physical connection |
| Hardware | Actual interface, MIDI, gain staging, grounding, and signal path | General correctness across all hosts and configurations |

A check that was not run must be reported as **not run**, not passed. A missing toolchain is a documented limitation, not a reason to delete the corresponding test.

## Review provenance

The review compared the frozen branch against the merged main baseline and inspected its changed-file inventory and relevant Python/Cmajor/documentation changes. The branch is intentionally left untouched. This document is a decision record of what may be learned from it, not an endorsement of its implementation.
