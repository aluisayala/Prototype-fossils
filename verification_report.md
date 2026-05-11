# OPHI State Space Verification Report

Run generated from explicit parameters.

## Command Parameters

```bash
python ophi_state_space_ai.py \
  --steps 200 \
  --seed 987654321 \
  --initial 0.731283 \
  --bias -0.00371 \
  --alpha 1.00297 \
  --reliability 0.9981 \
  --grounding 1.0 \
  --size 1024 \
  --out outputs_random_run
```

## Operator

`omega = (state + bias) * alpha * reliability * grounding`

## SE44 Thresholds

- Coherence C >= 0.985
- Entropy S <= 0.01
- RMS Drift <= 0.001

## Summary

- Run ID: `718c861d09c74f4ca7441a30d1c956efa7b9eba6f58aed5cce642a50530ec560`
- Initial state: `0.731283000000`
- Final state: `0.502498461013`
- Minimum state: `0.502235097394`
- Maximum state: `0.731283000000`
- Accepted steps: `195`
- Rejected steps: `5`
- Mean coherence: `0.990756012797`
- Max entropy: `0.020000000000`
- Max RMS drift: `0.130802837479`
- Merkle root: `a53c4bd03df5667fb85331914ebc5448c306a7f5cd32e0539ffad130f61517d9`

## Generated Artifacts

1. `01_trajectory.png`
2. `02_phase_portrait.png`
3. `03_density_heatmap.png`
4. `04_rgb_state_space_image.png`
5. `ophi_state_space_ledger.json`
6. `ophi_state_space_ledger.csv`
7. `mutable_shell_rejections.json`
8. `verification_report.md`

## Verification Note

This output set is generated deterministically from the supplied parameters in this notebook environment. If a separate canonical `ophi_state_space_ai.py` implementation exists, byte-identical reproducibility requires running that exact source file and comparing ledger hashes.
