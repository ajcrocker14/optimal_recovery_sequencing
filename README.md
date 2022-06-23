# Optimal Recovery Sequencing

## Getting Started

These instructions will get you a copy of the project up and running on your local machine. This project is currently optimized for and tested in Ubuntu. The following instructions are for configuring the project and dependent github repositories in Ubuntu.

This project relies on a C implementation of Algorithm-B for repeatedly solving the traffic assignment problem (Robert Dial), implemented by Dr. Steve Boyles and parallelized by Rishabh Thakkar and Karthik Velayutham, located at https://github.com/spartalab/tap-b as part of UT Austin's SPARTA lab.

For testing and research, this project makes use of the Transportation Networks for Research repository at https://github.com/bstabler/TransportationNetworks. For licensing requirements, please see each respective github for up to date information.

### Required dependencies

Python3 with the following libraries: pickle, sys, traceback, os, os.path, copy, shlex, subprocess, shutil, pdb, pandas, matplotlib, json, random, operator, functools, itertools, prettytable, operator, miltiprocessing, graphing, scipy, math, argparse, numpy, cProfile, pstats, networkx, ctypes, time, correspondence, progressbar, network, geopy, sklearn, tensorflow.

If on Windows OS, you will need WSL 1 and Ubuntu, or another method of operating in a Linux environment. If on Mac OS, you will also need a Linux environment to run the program as currently published.

C-compiler (we used gcc), installed through Ubuntu.

### Configurations

The below installation instructions will result in a working copy of the project on your local machine using Ubuntu without modifications to the underlying code. If you desire to install TAP-B and Transportation Networks repositories outside the current project folder (optimal_recovery_sequencing), some modifications to code files would be required. If operating directly on Mac OS, Windows OS, or a different Linux distribution, code modifications will be required in order to run the program. In the future, we hope to update this implementation, and add installation/modification instructions for Mac OS and Windows OS, but these are not included in the current iteration.

### Installation/Usage

In Ubuntu in your desired working directory, first, clone the required repos as such:

```
git clone https://github.com/cangokalp/optimal_recovery_sequencing.git

cd optimal_recovery_sequencing

git clone https://github.com/spartalab/tap-b.git

git clone https://github.com/bstabler/TransportationNetworks.git

```

To build an executable of TAP-B, in tap-b directory run 'make' in Ubuntu.

## Authors

* **Can Gokalp** - *Primary implementation* 
* **Abigail Crocker** - *Updated implementation for multi-class demand and multiple work crews*

## License

This project is licensed under the ___ license - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgements
