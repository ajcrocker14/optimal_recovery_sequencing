#!/bin/bash
net_name="Berlin-Mitte-Center"
echo $net_name
broken=8
reps=20

cd saved/TransportationNetworks/$net_name
rm -r $broken

cd ~/optimal_recovery_sequencing

python3 find_sequence.py -n $net_name -b $broken -l 1 -r $reps --sa 1 2 -s -g True

cd saved/TransportationNetworks/$net_name/$broken
cat */results.csv > combinedResults.csv
#cat */damaged_attributes.csv > attributes.csv

cd ~/optimal_recovery_sequencing/saved/TransportationNetworks/$net_name
mv $broken/combinedResults.csv  saResults$broken.csv
#mv $broken/attributes.csv  attributes$broken.csv

cd ~/optimal_recovery_sequencing
