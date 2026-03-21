from __future__ import annotations

import argparse

from . import sample, train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="littlelm", description="LittleLM command entrypoint")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Build n-gram dictionary files")
    train_parser.set_defaults(handler=train.main)

    sample_parser = subparsers.add_parser("sample", help="Generate text from n-gram dictionaries")
    sample_parser.set_defaults(handler=sample.main)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, remaining = parser.parse_known_args(argv)
    handler = args.handler
    return handler(remaining)


if __name__ == "__main__":
    raise SystemExit(main())
