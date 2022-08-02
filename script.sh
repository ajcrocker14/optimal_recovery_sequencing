#!/bin/bash
net_name="SiouxFalls"
echo $net_name
broken=16
reps=10

cd saved/TransportationNetworks/$net_name
rm -r $broken

cd ~/optimal_recovery_sequencing

python3 find_sequence.py -n $net_name -b $broken -l 1 -r $reps -a True --sa True

cd saved/TransportationNetworks/$net_name/$broken
cat */results.csv > combinedResults.csv

cd ~/optimal_recovery_sequencing/saved/TransportationNetworks/$net_name
mv $broken/combinedResults.csv  approx2Results$broken.csv

cd ~/optimal_recovery_sequencing
