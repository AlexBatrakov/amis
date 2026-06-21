"""Command-line entry point for AMIS."""

import argparse
from collections.abc import Sequence

READY_MESSAGE = "AMIS repository foundation is ready."


def main(argv: Sequence[str] | None = None) -> int:
    """Run the minimal AMIS command-line interface."""
    parser = argparse.ArgumentParser(
        prog="amis",
        description="Ankh-Morpork Intelligence System",
    )
    parser.parse_args(argv)
    print(READY_MESSAGE)
    return 0
