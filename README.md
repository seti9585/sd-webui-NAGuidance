# sd-webui-NAGuidance

[Normalized Attention Guidance (NAG)](https://arxiv.org/abs/2505.21179) を
Forge 系 WebUI（reForge / Forge Classic / Forge Neo）向け拡張機能として移植したものです。

ComfyUI ビルトインノード `NAGuidance`（`comfy_extras/nodes_nag.py`）をベースにしています。

---

## 概要

NAG は self-attention（attn1）の **出力空間** でガイダンスを行うため、

- ネガティブプロンプトが効きにくい **蒸留モデル**（Turbo / Lightning / Hyper）での負方向制御を復元できます
- **少ステップサンプリング** での品質向上にも有効です
- **UNet 系モデル（SDXL / SD1.5 等）** で動作します
- 既存の Pre-CFG / Post-CFG 系拡張（TCFG / SkimmedCFG / MaHiRo / FreSca）と **独立した軸**で動作し、それらと同時に使用できます

---

## インストール

Forge の拡張機能インストール UI から URL を入力：

**Extensions → Install from URL:**

```
https://github.com/seti9585/sd-webui-NAGuidance
```

---

## パラメータ

| パラメータ | デフォルト | 範囲 | 説明 |
|---|---|---|---|
| **NAG Scale** | 5.0 | 0.0 〜 50.0 | ガイダンス強度。CFG スケールに相当。大きいほどネガティブプロンプトの影響が強くなる |
| **NAG Alpha** | 0.5 | 0.0 〜 1.0 | 元の attention 出力との混合比。1.0 で完全置換、0.0 で無効と同等 |
| **NAG Tau**   | 1.5 | 1.0 〜 10.0 | L1 ノルム比のクリップ上限。小さいほど逸脱を強く抑制。アーティファクト抑止に有効 |

### 推奨設定

パラメータの効き方は環境・モデルへの依存が大きいため、弱めの値から始めて調整することを推奨します。

| ユースケース | Scale | Alpha | Tau |
|---|---|---|---|
| 蒸留モデル（Turbo / Lightning） | 5.0 〜 10.0 | 0.5 〜 0.75 | 1.5 〜 2.5 |
| SDXL（通常ステップ数） | 1.0 〜 3.0 | 0.1 〜 0.3 | 2.0 〜 3.0 |

---

## アルゴリズム

```
# self-attention (attn1) 出力への介入

z_pos  = attn1出力のうち cond 側
z_neg  = attn1出力のうち uncond 側

# CFG と同形の外挿（attention 空間）
guided = z_pos * nag_scale - z_neg * (nag_scale - 1)

# L1 ノルム正規化 + τ クリップ（マニフォールドからの逸脱を防ぐ）
ratio        = L1norm(guided) / L1norm(z_pos)
scale_factor = min(ratio, nag_tau) / ratio
guided_norm  = guided * scale_factor

# α-blend で元の attention 出力と混合
z_final = guided_norm * nag_alpha + z_pos * (1 - nag_alpha)

# cond スロットを z_final で上書き
```

---

## 他拡張機能との共存

| 拡張機能 | 共存 | 備考 |
|---|---|---|
| sd-webui-TCFG | ✅ | Pre-CFG 軸。独立して動作 |
| sd-webui-SkimmedCFG | ✅ | Pre-CFG 軸。独立して動作 |
| sd-webui-MaHiRo | ✅ | Post-CFG 軸。独立して動作 |
| sd-webui-FreSca | ✅ | Post-CFG 軸。独立して動作 |
| sd-webui-DifferentialDiffusion | ✅ | denoise_mask 軸。独立して動作 |

---

## 動作確認環境

| 環境 | 状態 |
|---|---|
| reForge + SDXL | ✅ 確認済み |
| Forge Neo + SDXL | ✅ 確認済み |
| Forge Neo + Anima (DiT) | ❌ 現時点では効果なし |
| A1111 | ❌ 非対応 |

> **Anima について:** NAG は UNet の `attn1_output_patch` フック経由で動作します。Anima(NextDiT) はアーキテクチャが異なるため、現時点ではこのフックによるガイダンスが画像に反映されません。ComfyUI 本家も Anima には未対応であり、対応方針が固まった段階で追従する予定です。

---

## 既知の制限

- **DiT 系モデル（Anima / Flux 等）は未対応**。`attn1_output_patch` は UNet 経路でのみ発火します。
- **SDXL ではデフォルト値（Scale=5.0）は強すぎる**場合があります。Scale=1.0〜2.0 程度から始めることを推奨します。

---

## ライセンス

MIT License

---

## 謝辞

- 原論文: [Normalized Attention Guidance (arXiv:2505.21179)](https://arxiv.org/abs/2505.21179)
- ComfyUI 実装: [Comfy-Org/ComfyUI](https://github.com/Comfy-Org/ComfyUI) `comfy_extras/nodes_nag.py`
