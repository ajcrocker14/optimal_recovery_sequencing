# Optimal Recovery Sequencing

## Getting Started

These instructions will get you a copy of the project up and running on your local machine. This project is currently optimized for and tested in Ubuntu. The following instructions are for configuring the project and dependent github repositories in Ubuntu.

This project relies on a C implementation of Algorithm-B for repeatedly solving the traffic assignment problem (Robert Dial), implemented by Dr. Steve Boyles and parallelized by Rishabh Thakkar and Karthik Velayutham, located at https://github.com/spartalab/tap-b as part of UT Austin's SPARTA lab.

For testing and research, this project makes use of the Transportation Networks for Research repository at https://github.com/bstabler/TransportationNetworks. For licensing requirements, please see each respective github for up to date information.

### Required dependencies

Python3 with the following libraries: pandas, numpy, scipy, geopy. In order to run options using machine learning you would need tensorflow (requires WSL 2 not WSL 1 to use with cuda).

If on Windows OS, you will need WSL and Ubuntu, or another method of operating in a Linux environment. If on Mac OS, you will also need a Linux environment to run the program as currently published.

C-compiler (we used gcc), installed through Ubuntu.

### Configurations

The below installation instructions will result in a working copy of the project on your local machine using Ubuntu without modifications to the underlying code. If you desire to install TAP-B and Transportation Networks repositories outside the current project folder (optimal_recovery_sequencing), some modifications to code files would be required. If operating directly on Mac OS, Windows OS, or a different Linux distribution, code modifications will be required in order to run the program. In the future, we hope to update this implementation, and add installation/modification instructions for Mac OS and Windows OS, but these are not included in the current iteration.

### Installation/Usage

In Ubuntu in your desired working directory, first, clone the required repos as such:

```
git clone https://github.com/ajcrocker14/optimal_recovery_sequencing.git

cd optimal_recovery_sequencing

git clone https://github.com/ajcrocker14/tap-b.git

git clone https://github.com/ajcrocker14/TransportationNetworks.git

```

To build an executable of TAP-B, in tap-b directory (tui branch) run 'make' in Ubuntu.

Test script for optimal recovery sequencing after completing the above:
```
python3 find_sequence.py -n SiouxFalls -b 5 -r 1
```

## Authors

* **Can Gokalp** - *Original implementation* 
* **Abigail Crocker** - *Updated implementation with improved exact optimal sequence implementation, improved beam search, simulated annealing, additional estimation and MIP methods, multi-class demand, and multiple work crews*

## License

Portions of this project are licensed under the MIT license - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgements

This research was supported by the U.S. Army Advanced Civil Schooling program. The views expressed herein are those of the authors and do not necessarily reflect the position of the Department of the Army or the Department of Defense.

