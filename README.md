# Spot Reinforcement Learning MSc

## Overview

This repository contains all the code that was used for our master thesis. It is based off of the template repository for creating isaac lab projects.


## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). We recommend using the conda installation as it simplifies calling Python scripts from the terminal.

- Clone this repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

- Using a python interpreter that has Isaac Lab installed, install the library

```bash
python -m pip install -e source/spot_stuff
```

- Verify that the extension is correctly installed by running the following command:

```bash
python scripts/rsl_rl/train.py --task=Msc-v0
```
