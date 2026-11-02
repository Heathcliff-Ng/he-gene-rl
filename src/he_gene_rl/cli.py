"""Command-line entry points; training and evaluation are separate operations."""
from __future__ import annotations

import argparse
import json

from . import __version__
from .workflow import METHODS


def parser():
    result = argparse.ArgumentParser(prog="he-gene-rl", description=__doc__)
    result.add_argument("--version", action="version", version=__version__)
    commands = result.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Train a predictor, adapt it, and train PPO")
    train.add_argument("--config", required=True)
    train.add_argument("--data", required=True, help="Portable JSON data manifest")
    train.add_argument("--output", required=True)
    train.add_argument("--stage", choices=("pretrain", "adapt", "ppo", "all"), default="all")
    train.add_argument("--device", default=None, help="Override config: cpu, cuda, or cuda:N")

    predict = commands.add_parser("predict", help="Image-only inference; never load expression targets")
    predict.add_argument("--run", required=True)
    predict.add_argument("--data", required=True)
    predict.add_argument("--source", required=True)
    predict.add_argument("--target", required=True)
    predict.add_argument("--seed", type=int, required=True)
    predict.add_argument("--split", default="test")
    predict.add_argument("--method", choices=METHODS, default="ppo_sequence")
    predict.add_argument("--batch-size", type=int, default=512)
    predict.add_argument("--device", default="cpu")
    predict.add_argument("--output", required=True)

    for name in ("evaluate", "compare"):
        command = commands.add_parser(name, help="Score completed predictions using expression targets")
        command.add_argument("--data", required=True)
        command.add_argument("--target", required=True)
        command.add_argument("--split", default="test")
        command.add_argument("--neighbors", type=int, default=8)
        command.add_argument("--output", required=True)
        if name == "evaluate":
            command.add_argument("--predictions", required=True)
        else:
            command.add_argument("--candidate", required=True)
            command.add_argument("--reference", required=True)
            command.add_argument("--bootstrap-replicates", type=int, default=2000)

    evaluation = commands.add_parser("evaluate-run", help="Evaluate frozen models and matched controls")
    evaluation.add_argument("--run", required=True)
    evaluation.add_argument("--data", required=True)
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--device", default="cpu")
    evaluation.add_argument("--split", default="test")
    evaluation.add_argument("--methods", nargs="+", choices=METHODS, default=None)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "train":
        from .workflow import train
        train(args.config, args.data, args.output, args.stage, args.device)
    elif args.command == "predict":
        from .workflow import predict
        predict(args.run, args.data, args.source, args.target, args.seed, args.split,
                args.output, args.device, args.method, args.batch_size)
    elif args.command == "evaluate":
        from .evaluation import evaluate
        report = evaluate(args.predictions, args.data, args.target, args.split, args.output, args.neighbors)
        print(json.dumps(report, indent=2))
    elif args.command == "compare":
        from .evaluation import compare
        report = compare(args.candidate, args.reference, args.data, args.target, args.split,
                         args.output, args.neighbors, args.bootstrap_replicates)
        print(json.dumps(report, indent=2))
    else:
        from .evaluation import evaluate_run
        evaluate_run(args.run, args.data, args.output, args.device, args.methods, args.split)


if __name__ == "__main__":
    main()
