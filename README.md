# sd-webui-NAGuidance

**EN** | [日本語](#日本語)

Attention-level guidance extension for Stable Diffusion WebUI (Forge-based).  
Restores negative guidance by extrapolating in the self-attention output space,  
independent of the CFG loop — so it works even on distilled / low-CFG sampling.

Paper: [arXiv:2505.21179](https://arxiv.org/abs/2505.21179)  
Original implementation: ComfyUI built-in [`comfy_extras/nodes_nag.py`](https://github.com/comfyanonymous/ComfyUI/blob/master/comfy_extras/nodes_nag.py)

> Operates on `attn1` output, a separate axis from Pre-CFG / Post-CFG extensions (TCFG / SkimmedCFG / MaHiRo / FreSca). It can be combined with all of them.

---

## Installation

**Extensions → Install from URL:**

```
https://github.com/seti9585/sd-webui-NAGuidance
```

---

## Parameters

| Parameter | Default | Range | Description |
| --------- | ------- | ----- | ----------- |
| **NAG Scale** | 5.0 | 0.0 – 50.0 | Guidance strength. Analogous to the CFG scale. |
| **NAG Alpha** | 0.5 | 0.0 – 1.0 | Blend ratio between normalized guided output and the original attention output. `1.0` = full replacement. |
| **NAG Tau**   | 1.5 | 1.0 – 10.0 | Upper clip bound on the L1-norm ratio. Controls how strongly manifold deviation is suppressed. |

The defaults (`5.0 / 0.5 / 1.5`) are a reasonable starting point for most models.

---

## When to use

NAG restores negative-prompt influence where classifier-free guidance is weak or absent:

- **Distilled / few-step models** (Turbo / Lightning / Hyper) where negative prompts barely take effect
- **Low-CFG or CFG-disabled sampling**, where there is no negative branch for CFG to use

---

## Algorithm

```
# Applied to the self-attention (attn1) output

z_pos  = attn1 output for the cond   branch
z_neg  = attn1 output for the uncond branch

guided = z_pos × nag_scale − z_neg × (nag_scale − 1)   # CFG-style extrapolation

ratio        = L1norm(guided) / L1norm(z_pos)
scale_factor = min(ratio, nag_tau) / ratio             # L1-norm clip
guided_norm  = guided × scale_factor

z_final = guided_norm × nag_alpha + z_pos × (1 − nag_alpha)   # α-blend

→ z_final overwrites the cond / uncond attention slots
```

Guidance is extrapolated in attention space, normalized by the L1-norm ratio and clipped at `nag_tau` to prevent deviation from the manifold, then blended back with `nag_alpha`.

### Hook

```python
unet.set_model_attn1_output_patch(nag_attention_output_patch)
unet.model_options["disable_cfg1_optimization"] = True
```

NAG forces `disable_cfg1_optimization = True` so the unconditional branch is always evaluated (NAG needs both cond and uncond attention outputs).

---

## Architecture independence

Because NAG operates on **attention outputs**, it does not depend on the epsilon (UNet) or velocity (flow-matching) prediction space.  
It works on both **UNet models (SDXL etc.)** and **flow-matching DiT models (Anima etc.)** without modification.  
For DiT models that split img / txt tokens, the `img_slice` branch handles the image-token region.

---

## Tested environments

- reForge (Python 3.10) — SDXL-family models
- Forge Neo (Python 3.12) — including Anima, and txt2img + Hires.fix

Not compatible with A1111 (`forge_objects` backend required).

---
---

# 日本語

**[English](#sd-webui-naguidance)** | 日本語

Forge 系 WebUI 向け attention 層ガイダンス拡張機能。  
self-attention 出力空間で外挿を行うことでネガティブガイダンスを復元します。  
CFG ループから独立しているため、蒸留モデルや低 CFG サンプリングでも機能します。

論文: [arXiv:2505.21179](https://arxiv.org/abs/2505.21179)  
原実装: ComfyUI ビルトイン [`comfy_extras/nodes_nag.py`](https://github.com/comfyanonymous/ComfyUI/blob/master/comfy_extras/nodes_nag.py)

> `attn1` 出力に作用し、Pre-CFG / Post-CFG 系拡張（TCFG / SkimmedCFG / MaHiRo / FreSca）とは別軸で動作します。これらすべてと併用可能です。

---

## インストール

**Extensions → Install from URL:**

```
https://github.com/seti9585/sd-webui-NAGuidance
```

---

## パラメータ

| パラメータ | 既定値 | 範囲 | 説明 |
| --- | --- | --- | --- |
| **NAG Scale** | 5.0 | 0.0 〜 50.0 | ガイダンス強度。CFG スケールに相当。 |
| **NAG Alpha** | 0.5 | 0.0 〜 1.0 | 正規化済み guided と元の attention 出力のブレンド比。`1.0` で完全置換。 |
| **NAG Tau**   | 1.5 | 1.0 〜 10.0 | L1 ノルム比のクリップ上限。マニフォールドからの逸脱抑制の強さ。 |

既定値（`5.0 / 0.5 / 1.5`）はほとんどのモデルで適切な出発点です。

---

## 使いどころ

NAG は CFG が弱い、または機能しない状況でネガティブプロンプトの効果を復元します。

- **蒸留・少ステップモデル**（Turbo / Lightning / Hyper）でネガティブプロンプトがほとんど効かない場合
- **低 CFG・CFG 無効**のサンプリングで、CFG が使うネガティブ側の枝が存在しない場合

---

## アルゴリズム

```
# self-attention（attn1）出力に対して適用

z_pos  = attn1 出力のうち cond   側
z_neg  = attn1 出力のうち uncond 側

guided = z_pos × nag_scale − z_neg × (nag_scale − 1)   # CFG と同形の外挿

ratio        = L1norm(guided) / L1norm(z_pos)
scale_factor = min(ratio, nag_tau) / ratio             # L1 ノルムクリップ
guided_norm  = guided × scale_factor

z_final = guided_norm × nag_alpha + z_pos × (1 − nag_alpha)   # α ブレンド

→ z_final で cond / uncond の attention スロットを上書き
```

attention 空間でガイダンスを外挿し、L1 ノルム比で正規化したうえで `nag_tau` でクリップしてマニフォールドからの逸脱を防ぎ、`nag_alpha` で元の出力と混合します。

### フック

```python
unet.set_model_attn1_output_patch(nag_attention_output_patch)
unet.model_options["disable_cfg1_optimization"] = True
```

NAG は cond / uncond 両方の attention 出力を必要とするため、`disable_cfg1_optimization = True` を設定して uncond の枝が常に評価されるようにします。

---

## アーキテクチャ非依存性

NAG は **attention 出力**に作用するため、epsilon（UNet）／ velocity（フローマッチング）の予測空間に依存しません。  
**UNet モデル（SDXL 等）**と**フローマッチング系 DiT モデル（Anima 等）**のどちらでも、変更なしで動作します。  
img / txt トークンを分割する DiT モデルに対しては、`img_slice` 分岐が画像トークン領域を処理します。

---

## 動作確認環境

- reForge（Python 3.10）— SDXL 系モデル
- Forge Neo（Python 3.12）— Anima を含む。txt2img + Hires.fix も確認

A1111 非対応（`forge_objects` バックエンドが必要）。

---

## ライセンス

MIT License — Based on: [arXiv:2505.21179](https://arxiv.org/abs/2505.21179)  
Original implementation: ComfyUI built-in `comfy_extras/nodes_nag.py`
