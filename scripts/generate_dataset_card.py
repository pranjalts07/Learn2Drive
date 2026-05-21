#!/usr/bin/env python3
"""Generate dataset card."""

import sys
from autonomy_stack.data_engine.cards import DatasetCard


def main():
    """Generate dataset card."""
    card = DatasetCard("synthetic_dataset", "0.1.0")
    output_path = "artifacts/dataset_card.json"

    import os
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    card.save(output_path)
    print(f"✓ Dataset card saved to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
