"""Command-line entry points; planning and calibration require no model download."""
import argparse
import json

from .evaluate import evaluate
from .experiment import design, generate, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="Print planned workload without downloading anything")
    plan.add_argument("--config", required=True)
    gen = commands.add_parser("generate", help="Generate a resumable paired experiment")
    gen.add_argument("--config", required=True)
    gen.add_argument("--out", required=True)
    gen.add_argument("--backend", choices=["synthetic", "audioldm2"], required=True)
    gen.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    gen.add_argument("--offload", action="store_true")
    ev = commands.add_parser("evaluate", help="Verify, measure, compare and build a listening report")
    ev.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "plan":
        cfg = load_config(args.config)
        count = sum(1 for _ in design(cfg))
        print(json.dumps({"model": cfg["model_id"], "prompt_pairs": len(cfg["pairs"]),
                          "clips": count, "total_audio_seconds": count*cfg["duration_s"],
                          "steps_per_clip": cfg["steps"]}, indent=2))
    elif args.command == "generate":
        generate(args.config, args.out, args.backend, args.device, args.offload)
    else:
        evaluate(args.out)


if __name__ == "__main__":
    main()
