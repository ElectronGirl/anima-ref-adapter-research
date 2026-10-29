"""Write workflows/anima_ref_adapter_advanced.json: the Loader + Ref Latent + Ref Guider graph, core nodes only.

  python deploy/make_advanced_workflow.py --out workflows/anima_ref_adapter_advanced.json
"""
import argparse
import json
from pathlib import Path

from make_simple_workflow import NEGATIVE, POSITIVE
from make_test_workflow import NODES, to_ui

NOTE = """## Anima Reference Adapter (advanced)

The separate nodes, for the pilot adapter and for options the one-node workflow does not expose.

- **Anima Ref Latent** fits the reference into the output size (white letterbox) and returns the empty latent for that size. The pilot needs the reference at the output size; v2 reads any size.
- **Anima Ref Guider** (`SamplerCustomAdvanced`): `cfg`, `ref_guidance`, `correction`, and the window `ref_start` / `ref_end` with `outside` = keep the reference without extra guidance, or drop it.
- **Anima Ref Adapter Loader**: keep `ref_offset_h` at 1.0 (training value).
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lora", default="anima_ref_adapter_v2_r64.safetensors")
    args = parser.parse_args()
    widgets = {"UNETLoader": ["anima-base-v1.0.safetensors", "default"], "VAELoader": ["qwen_image_vae.safetensors"],
               "AnimaRefAdapterLoader": [args.lora, 1.0, 1.0], "LoadImage": ["reference.png", "image"],
               "KSamplerSelect": ["er_sde"]}
    nodes = []
    for nid, ntype, title, w, inputs, outputs, pos, size in NODES:
        if ntype in ("ModelComputeDtype", "PathchSageAttentionKJ"):
            continue
        inputs = {k: (4, 0, "MODEL") if v[0] in (5, 6) else v for k, v in inputs.items()}
        if nid == 9:
            w = [POSITIVE]
        elif nid == 10:
            w = [NEGATIVE]
        nodes.append((nid, ntype, title, widgets.get(ntype, w), inputs, outputs, pos, size))
    workflow = to_ui(nodes, NOTE)
    workflow["nodes"][-1]["title"] = "How to use"
    args.out.write_text(json.dumps(workflow, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
