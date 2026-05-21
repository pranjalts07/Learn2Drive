#!/usr/bin/env python3
"""Generate model card."""

import sys
import json
from autonomy_stack.data_engine.cards import ModelCard


def main():
    """Generate model card."""
    card = ModelCard("behavior_cloning_v1", "0.1.0")
    output_path = "artifacts/model_card.json"

    import os
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    card.save(output_path)
    print(f"✓ Model card saved to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
