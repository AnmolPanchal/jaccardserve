"""Download models needed for Phases 2-3."""
import argparse
from huggingface_hub import snapshot_download

MODELS = {
    "gen": "Qwen/Qwen2.5-3B-Instruct",
    "judge": "Qwen/Qwen2.5-7B-Instruct",
    "fallback_gen": "Qwen/Qwen2.5-1.5B-Instruct",
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["gen"],
                    choices=list(MODELS.keys()),
                    help="Which models to download")
    args = ap.parse_args()

    for key in args.models:
        model_id = MODELS[key]
        print(f"Downloading {model_id} ...")
        path = snapshot_download(model_id)
        print(f"  -> {path}")

if __name__ == "__main__":
    main()
