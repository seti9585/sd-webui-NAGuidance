# sd-webui-NAGuidance

**Normalized Attention Guidance (NAG)** for Forge-derived Stable Diffusion WebUIs.

A port of the ComfyUI built-in `NAGuidance` node (`comfy_extras/nodes_nag.py`) to the Forge extension format (reForge / Forge Classic / Forge Neo).

Paper: [Normalized Attention Guidance (arXiv:2505.21179)](https://arxiv.org/abs/2505.21179)

---

## What it does

NAG restores **negative guidance** by operating directly in the self-attention output space, rather than through the CFG loop. This makes it useful where classifier-free guidance is weak or unavailable:

- **Distilled / few-step models** (Turbo / Lightning / Hyper) where negative prompts barely take effect
- **Low-CFG or CFG-disabled sampling**, where there is no negative branch for CFG to use

Because NAG works on attention outputs, it is **architecture-agnostic**: it works on both UNet models (SDXL etc.) and flow-matching DiT models (Anima etc.), since it does not depend on the epsilon / velocity prediction space.

It also operates on an **independent axis** from CFG-based guidance extensions (TCFG / SkimmedCFG / MaHiRo / FreSca), so it can be combined with them.

---

## Compatibility

| WebUI | Status |
|---|---|
| reForge | ✅ Confirmed working |
| Forge Neo | ✅ Confirmed working |
| Forge Classic | Expected to work (same hook API) |
| A1111 | ❌ Not supported (no Forge backend) |

Confirmed on both UNet (SDXL, e.g. Illustrious-based) and DiT (Anima-based) models.

---

## Installation

### From WebUI

1. Open the **Extensions** tab → **Install from URL**
2. Paste:
   ```
   https://github.com/seti9585/sd-webui-NAGuidance
   ```
3. Click **Install**, then **Apply and restart UI**

### Manual

Clone into your WebUI's `extensions` folder:

```bash
cd stable-diffusion-webui/extensions
git clone https://github.com/seti9585/sd-webui-NAGuidance
```

---

## Usage

1. Open the **NAGuidance (Normalized Attention Guidance)** accordion in the txt2img / img2img panel
2. Check **Enable NAGuidance**
3. Adjust the parameters and generate

The default values (`5.0 / 0.5 / 1.5`) are a reasonable starting point for most models.

---

## Parameters

| Parameter | Default | Range | Description |
|---|---|---|---|
| **NAG Scale** | 5.0 | 0.0 – 50.0 | Guidance strength. Analogous to the CFG scale. Higher = stronger negative guidance. |
| **NAG Alpha** | 0.5 | 0.0 – 1.0 | Blend ratio between the normalized guided output and the original attention output. `1.0` = full replacement. |
| **NAG Tau** | 1.5 | 1.0 – 10.0 | Upper clip bound on the L1-norm ratio. Controls how strongly deviation from the manifold is suppressed. |

---

## How it works

NAG is applied to the output of the self-attention (`attn1`) layer:

```
z_pos  = attn1 output for the cond  branch
z_neg  = attn1 output for the uncond branch

# CFG-style extrapolation, in attention space
guided = z_pos * nag_scale - z_neg * (nag_scale - 1)

# L1-norm normalization + tau clip (prevents manifold deviation)
ratio        = L1norm(guided) / L1norm(z_pos)
scale_factor = min(ratio, nag_tau) / ratio
guided_norm  = guided * scale_factor

# alpha-blend with the original attention output
z_final = guided_norm * nag_alpha + z_pos * (1 - nag_alpha)
```

The result overwrites the attention output, and CFG-1 optimization is disabled so the uncond branch is always evaluated.

### Hook

NAGuidance registers on the self-attention output patch, completely independent of the Pre-CFG / Post-CFG hooks used by other guidance extensions:

```python
unet.set_model_attn1_output_patch(nag_attention_output_patch)
unet.model_options["disable_cfg1_optimization"] = True
```

---

## Compatibility with other extensions

NAG runs on its own axis (the attention layer, outside the CFG loop), so it composes with the author's other Forge guidance extensions:

| Extension | Hook | Axis |
|---|---|---|
| [sd-webui-TCFG](https://github.com/seti9585/sd-webui-TCFG) | Pre-CFG / Post-CFG | CFG |
| [sd-webui-SkimmedCFG](https://github.com/seti9585/sd-webui-SkimmedCFG) | Pre-CFG | CFG |
| [sd-webui-MaHiRo](https://github.com/seti9585/sd-webui-MaHiRo) | Post-CFG | CFG |
| [sd-webui-FreSca](https://github.com/seti9585/sd-webui-FreSca) | Post-CFG / frequency | CFG |
| **sd-webui-NAGuidance** | **attn1 output** | **attention (independent)** |

> **Note:** When stacking several CFG-axis extensions at very high CFG values (e.g. CFG ≈ 24) on a UNet backend, the combined per-hook corrections can accumulate and degrade the image. NAG itself is not the cause (it is on a separate axis), but if you stack many guidance extensions, keeping CFG in a moderate range (≈7–10) is recommended.

---

## Notes

- NAG forces `disable_cfg1_optimization = True`, so the unconditional branch is always computed. This means a single NAG-enabled pass costs roughly the same as a standard CFG pass even at CFG 1.
- Parameters are written to the PNG metadata (`NAG Scale` / `NAG Alpha` / `NAG Tau`) for reproducibility.

---

## Credits

- Algorithm: *Normalized Attention Guidance* — [arXiv:2505.21179](https://arxiv.org/abs/2505.21179)
- Reference implementation: ComfyUI built-in `comfy_extras/nodes_nag.py`

---

## License

MIT — see [LICENSE](LICENSE).
