# Third-Party Notices

This project includes or references the following third-party software:

## CARLA Simulator

CARLA is an open-source simulator for autonomous driving research and development.

- **Project**: CARLA (https://carla.org/)
- **License**: CARLA is licensed under the CARLA Simulator Open License (CSOL)
- **Note**: This project provides optional integration with CARLA for data collection and evaluation. CARLA is only required if using the CARLA mode (default is synthetic mode, which does not require CARLA).

## Python Dependencies

This project uses the following Python packages (see requirements.txt for full list):
- NumPy
- PyTorch
- ONNX
- PyYAML
- pytest
- and others

These are installed via pip and are subject to their respective licenses.

## Attribution

If you use the synthetic mode or the behavior cloning/offline RL components, no additional attribution is required beyond this file and the project LICENSE.

If you extend this project to use CARLA, please follow CARLA's attribution requirements as specified in the CARLA documentation.
