"""
sd-webui-NAGuidance / scripts/sd_webui_nag.py

Integrates Normalized Attention Guidance into Forge-derived WebUIs.
Supported: reForge / Forge Classic / Forge Neo
"""

import os
import sys
import logging

import gradio as gr
from modules import scripts

# Reference the sd_webui_nag package next to scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sd_webui_nag import apply_nag

logger = logging.getLogger(__name__)


def _has_forge_backend(p) -> bool:
    return hasattr(p, "sd_model") and hasattr(p.sd_model, "forge_objects")


class NAGuidanceScript(scripts.Script):

    def __init__(self):
        self.enabled = False

    # ───────────── Forge metadata ─────────────

    def title(self):
        return "NAGuidance"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    # ───────────── UI ─────────────

    def ui(self, is_img2img):
        with gr.Accordion("NAGuidance (Normalized Attention Guidance)", open=False):
            enabled = gr.Checkbox(label="Enable NAGuidance", value=False)

            with gr.Row():
                nag_scale = gr.Slider(
                    minimum=0.0, maximum=50.0, step=0.1, value=5.0,
                    label="NAG Scale",
                    info="Guidance strength. Analogous to the CFG scale.",
                )
                nag_alpha = gr.Slider(
                    minimum=0.0, maximum=1.0, step=0.01, value=0.5,
                    label="NAG Alpha",
                    info="Blend ratio with the original attention. 1.0 = full replacement.",
                )
                nag_tau = gr.Slider(
                    minimum=1.0, maximum=10.0, step=0.01, value=1.5,
                    label="NAG Tau",
                    info="Upper clip bound on the L1 norm ratio. Controls deviation suppression.",
                )

        # Infotext round-trip (PNG Info -> Send to txt2img / img2img).
        # There is no dedicated enabled key; the three NAG keys are written only
        # when active, so the presence of "NAG Scale" means ON, absence OFF.
        # Enable must be a callable because infotext paste leaves a component
        # untouched when its key is absent; the callable resolves a missing key
        # to False, forcing OFF for faithful same-seed reproduction. The three
        # sliders use plain keys. The metadata write lives in process() (below).
        self.infotext_fields = [
            (enabled,   lambda d: "NAG Scale" in d),
            (nag_scale, "NAG Scale"),
            (nag_alpha, "NAG Alpha"),
            (nag_tau,   "NAG Tau"),
        ]

        return [enabled, nag_scale, nag_alpha, nag_tau]

    # ───────────── Effective configuration ─────────────

    def _resolve(self, args):
        enabled   = bool(args[0])  if len(args) >= 1 else False
        nag_scale = float(args[1]) if len(args) >= 2 else 5.0
        nag_alpha = float(args[2]) if len(args) >= 3 else 0.5
        nag_tau   = float(args[3]) if len(args) >= 4 else 1.5
        return enabled, nag_scale, nag_alpha, nag_tau

    # ───────────── Metadata write ─────────────

    def process(self, p, *args):
        # Write here (once, before the batch loop) so create_infotext captures
        # it for every saved image, which the PNG Info round-trip depends on.
        if len(args) < 4:
            return
        enabled, nag_scale, nag_alpha, nag_tau = self._resolve(args)
        if not enabled:
            return
        p.extra_generation_params.update({
            "NAG Scale": nag_scale,
            "NAG Alpha": nag_alpha,
            "NAG Tau": nag_tau,
        })

    # ───────────── Hook registration ─────────────

    def process_before_every_sampling(self, p, *args, **kwargs):
        """
        Forge hook called right before sampling starts.
        Registers the attn1_output patch on the UNet here.
        """
        if len(args) < 4:
            logger.warning("[NAGuidance] process_before_every_sampling: missing args")
            return

        enabled, nag_scale, nag_alpha, nag_tau = self._resolve(args)
        self.enabled = enabled

        if not enabled:
            return

        if not _has_forge_backend(p):
            logger.warning("[NAGuidance] Requires Forge backend.")
            return

        unet = p.sd_model.forge_objects.unet
        unet = apply_nag(unet, nag_scale=nag_scale, nag_alpha=nag_alpha, nag_tau=nag_tau)
        p.sd_model.forge_objects.unet = unet
        logger.debug("[NAGuidance] applied (scale=%.2f alpha=%.2f tau=%.2f)",
                     nag_scale, nag_alpha, nag_tau)
