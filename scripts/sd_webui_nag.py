"""
sd-webui-NAGuidance / scripts/sd_webui_nag.py

Normalized Attention Guidance を Forge 系 WebUI に統合する拡張スクリプト。
対応環境: reForge / Forge Classic / Forge Neo
"""

import gradio as gr
from modules import scripts

# scripts/ と同階層の sd_webui_nag パッケージを参照
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sd_webui_nag import apply_nag


class NAGuidanceScript(scripts.Script):

    # ───────────── Forge 必須メタデータ ─────────────

    def title(self):
        return "NAGuidance"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    # ───────────── UI 定義 ─────────────

    def ui(self, is_img2img):
        with gr.Accordion("NAGuidance (Normalized Attention Guidance)", open=False):
            enabled = gr.Checkbox(label="Enable NAGuidance", value=False)

            with gr.Row():
                nag_scale = gr.Slider(
                    minimum=0.0, maximum=50.0, step=0.1, value=5.0,
                    label="NAG Scale",
                    info="ガイダンス強度。CFG スケールに相当。",
                )
                nag_alpha = gr.Slider(
                    minimum=0.0, maximum=1.0, step=0.01, value=0.5,
                    label="NAG Alpha",
                    info="元 attention との混合比。1.0 で完全置換。",
                )
                nag_tau = gr.Slider(
                    minimum=1.0, maximum=10.0, step=0.01, value=1.5,
                    label="NAG Tau",
                    info="L1 ノルム比のクリップ上限。逸脱抑制の強さ。",
                )

        return [enabled, nag_scale, nag_alpha, nag_tau]

    # ───────────── フック登録 ─────────────

    def process_before_every_sampling(self, p, *args, **kwargs):
        """
        サンプリング開始直前に呼ばれる Forge フック。
        ここで UNet に attn1_output パッチを登録する。
        """
        enabled, nag_scale, nag_alpha, nag_tau = args[:4]

        if not enabled:
            return

        unet = p.sd_model.forge_objects.unet
        unet = apply_nag(unet, nag_scale=nag_scale, nag_alpha=nag_alpha, nag_tau=nag_tau)
        p.sd_model.forge_objects.unet = unet

        # PNG メタデータ / プロンプト情報に記録
        p.extra_generation_params.update({
            "NAG Scale": nag_scale,
            "NAG Alpha": nag_alpha,
            "NAG Tau": nag_tau,
        })
