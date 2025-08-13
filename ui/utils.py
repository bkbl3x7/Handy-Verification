from pathlib import Path
from typing import Dict, List, Any, Tuple
import json

import torch
from PIL import Image

from handnet import make_backbone, make_transforms


def scan_experiments(root: str) -> Dict[str, Dict[str, Dict[str, List[Path]]]]:
    tree: Dict[str, Dict[str, Dict[str, List[Path]]]] = {}
    root_path = Path(root)
    if not root_path.exists():
        return tree
    for exp_name in sorted([p for p in root_path.iterdir() if p.is_dir()]):
        bb_map: Dict[str, Dict[str, List[Path]]] = {}
        for bb in sorted([p for p in exp_name.iterdir() if p.is_dir()]):
            stage_map: Dict[str, List[Path]] = {}
            for stage in sorted([p for p in bb.iterdir() if p.is_dir()]):
                runs = sorted([p for p in stage.iterdir() if p.is_dir()], reverse=True)
                if runs:
                    stage_map[stage.name] = runs
            if stage_map:
                bb_map[bb.name] = stage_map
        if bb_map:
            tree[exp_name.name] = bb_map
    return tree


def load_run_info(run_dir: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {"path": str(run_dir)}
    mfp = run_dir / "metrics.json"
    afp = run_dir / "args.json"
    rfp = run_dir / "roc.png"
    sfp = run_dir / "scores.csv"
    bfp = run_dir / "best.pt"
    if mfp.exists():
        try:
            info["metrics"] = json.loads(mfp.read_text())
        except Exception:
            info["metrics"] = None
    if afp.exists():
        try:
            info["args"] = json.loads(afp.read_text())
        except Exception:
            info["args"] = None
    info["roc_path"] = str(rfp) if rfp.exists() else ""
    info["scores_path"] = str(sfp) if sfp.exists() else ""
    info["weights_path"] = str(bfp) if bfp.exists() else ""
    return info


def load_model_for_verify(weights_path: str, device: torch.device):
    state = torch.load(weights_path, map_location=device)
    args = state.get('args', {}) or {}
    backbone = args.get('backbone', 'resnet18')
    model = make_backbone(backbone, embed_dim=128).to(device)
    model.load_state_dict(state['model'])
    model.eval()
    return model, args


def embed_images(model, pil_images: List[Image.Image], img_size: int, device: torch.device):
    _, eval_tf = make_transforms(img_size, aug=False)
    import torch as _torch
    xs = [eval_tf(im.convert('RGB')) for im in pil_images]
    xb = _torch.stack(xs, 0).to(device)
    with _torch.no_grad():
        z = model(xb)
    return z


def scan_runs_flat(root: str = "experiments") -> List[Dict[str, Any]]:
    """Scan experiments/<run_slug>/ directories with a manifest.json.
    Returns a list of runs with keys: path, dataset, pipeline, backbone, seed, created_at, stages (list).
    """
    out: List[Dict[str, Any]] = []
    rdir = Path(root)
    if not rdir.exists():
        return out
    for run_dir in sorted([p for p in rdir.iterdir() if p.is_dir()]):
        mfp = run_dir / "manifest.json"
        if not mfp.exists():
            # best-effort: skip or infer later as needed
            continue
        try:
            manifest = json.loads(mfp.read_text())
            manifest["path"] = str(run_dir)
            out.append(manifest)
        except Exception:
            continue
    return out
