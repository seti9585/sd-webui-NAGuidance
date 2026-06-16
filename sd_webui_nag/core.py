"""
sd-webui-NAGuidance / core.py

Normalized Attention Guidance のコア実装。
ComfyUI ビルトイン comfy_extras/nodes_nag.py を Forge 向けに移植。
原論文: arXiv:2505.21179
"""

import torch


def make_nag_patch(nag_scale: float, nag_alpha: float, nag_tau: float):
    """
    attn1 出力パッチ関数を生成して返す。

    Parameters
    ----------
    nag_scale : float
        ガイダンス強度。CFG スケールに相当。
    nag_alpha : float
        正規化済み guided と元の attn 出力のブレンド比。1.0 で完全置換。
    nag_tau : float
        L1 ノルム比のクリップ上限。マニフォールドからの逸脱抑制。

    Returns
    -------
    Callable[[Tensor, dict], Tensor]
        unet.set_model_attn1_output_patch() に渡せるパッチ関数。
    """

    def nag_attention_output_patch(out: torch.Tensor, extra_options: dict) -> torch.Tensor:
        # cond / uncond の両バッチが存在するかチェック
        cond_or_uncond = extra_options.get("cond_or_uncond", None)
        if cond_or_uncond is None:
            return out
        if not (0 in cond_or_uncond and 1 in cond_or_uncond):
            # cond のみ（または uncond のみ）の場合は何もしない
            return out

        # Flux 系 DiT の img / txt トークン分割への対応
        # extra_options["img_slice"] = (img_start, img_end) の形式で渡される
        img_slice = extra_options.get("img_slice", None)

        if img_slice is not None:
            # img トークン部分だけを取り出して処理し、後で戻す
            orig_out = out
            out = out[:, img_slice[0]:img_slice[1]]

        batch_size = out.shape[0]
        half_size = batch_size // len(cond_or_uncond)

        # cond_or_uncond のリスト内での cond(0) / uncond(1) の位置を取得
        ind_pos = cond_or_uncond.index(0)   # cond
        ind_neg = cond_or_uncond.index(1)   # uncond

        z_pos = out[half_size * ind_pos : half_size * (ind_pos + 1)]
        z_neg = out[half_size * ind_neg : half_size * (ind_neg + 1)]

        # CFG と同形の外挿（attention 空間）
        guided = z_pos * nag_scale - z_neg * (nag_scale - 1.0)

        # L1 ノルム正規化 + τ クリップ
        eps = 1e-6
        norm_pos    = torch.norm(z_pos,   p=1, dim=-1, keepdim=True).clamp_min(eps)
        norm_guided = torch.norm(guided,  p=1, dim=-1, keepdim=True).clamp_min(eps)

        ratio        = norm_guided / norm_pos
        scale_factor = torch.minimum(ratio, torch.full_like(ratio, nag_tau)) / ratio

        guided_normalized = guided * scale_factor

        # α ブレンド
        z_final = guided_normalized * nag_alpha + z_pos * (1.0 - nag_alpha)

        # 結果を書き戻す
        if img_slice is not None:
            # img_slice ありの場合は cond / uncond 両スロットを上書き
            orig_out[half_size * ind_pos : half_size * (ind_pos + 1), img_slice[0]:img_slice[1]] = z_final
            orig_out[half_size * ind_neg : half_size * (ind_neg + 1), img_slice[0]:img_slice[1]] = z_final
            return orig_out
        else:
            # img_slice なし（UNet / 通常 DiT）: cond スロットのみ上書き
            out[half_size * ind_pos : half_size * (ind_pos + 1)] = z_final
            return out

    return nag_attention_output_patch


def apply_nag(unet, nag_scale: float, nag_alpha: float, nag_tau: float):
    """
    UNet オブジェクトに NAG パッチを適用して返す。

    Parameters
    ----------
    unet : ModelPatcher
        p.sd_model.forge_objects.unet から取得した UNet。
        本関数内で clone() を行うため、呼び出し側での事前クローンは不要。
    nag_scale, nag_alpha, nag_tau : float
        NAG パラメータ（make_nag_patch 参照）。

    Returns
    -------
    ModelPatcher
        パッチ適用済みの cloned UNet。
    """
    unet = unet.clone()

    patch = make_nag_patch(nag_scale, nag_alpha, nag_tau)
    unet.set_model_attn1_output_patch(patch)

    # NAG は cond / uncond の両バッチが必要なため CFG 最適化を無効化
    # （最適化が有効だと uncond 推論がスキップされる場合がある）
    unet.model_options["disable_cfg1_optimization"] = True

    return unet
