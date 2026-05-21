#!/usr/bin/env python3
"""Collect episodes from CARLA simulator (optional, requires CARLA)."""

import sys


def main():
    print("CARLA Episode Collection")
    print("This script requires CARLA simulator to be running.")
    print("Configure CARLA in configs/carla_*.yaml files.")
    print("Currently in synthetic-only mode.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
