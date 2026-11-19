"""04_run_model.py: run one model over the suite.

Plain-language summary: loads the model named in a configuration file, shows it
every question in its original and altered forms, and saves the raw answers to
results/<run name>/raw.jsonl. This is the step that needs a GPU for real models.

Run:  python code/04_run_model.py configs/qwen2_5_vl_7b.yaml
      python code/04_run_model.py configs/pilot_smolvlm_cpu.yaml   (small CPU pilot)
"""
import argparse
import json
import os
import platform
import sys
import time

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from groundcheck.models import load_adapter  # noqa: E402
from groundcheck.run import run_items, select_items  # noqa: E402
from groundcheck.suite import Suite  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--limit", type=int, default=None, help="override items per dataset")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.limit:
        cfg["limit_per_dataset"] = a.limit
    suite = Suite(ROOT)
    ids = select_items(suite, cfg.get("datasets", ["SLAKE", "VQA-RAD", "PathVQA", "MIMIC-CXR-VQA", "MedXpertQA-MM"]),
                       cfg.get("splits", ["cal", "test"]), cfg.get("limit_per_dataset"),
                       cfg.get("require_variant"))
    out = os.path.join(ROOT, "results", cfg["run_name"])
    os.makedirs(out, exist_ok=True)
    meta = dict(config=cfg, n_items=len(ids), started=time.strftime("%Y-%m-%d %H:%M:%S"),
                python=platform.python_version(), host=platform.platform())
    try:
        import torch
        import transformers
        meta.update(torch=torch.__version__, transformers=transformers.__version__,
                    cuda=torch.cuda.is_available())
    except ImportError:
        pass
    json.dump(meta, open(os.path.join(out, "run_meta.json"), "w"), indent=2)
    json.dump(ids, open(os.path.join(out, "item_ids.json"), "w"))
    print(f"{cfg['run_name']}: {len(ids)} items")
    adapter = load_adapter(cfg)
    run_items(adapter, suite, ids, os.path.join(out, "raw.jsonl"), cfg)
    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(meta, open(os.path.join(out, "run_meta.json"), "w"), indent=2)
    print("done:", out)


if __name__ == "__main__":
    main()
