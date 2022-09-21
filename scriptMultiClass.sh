#!/bin/bash
net_name="Berlin-Mitte-Center2"
echo $net_name
broken=8
reps=1

cd saved/TransportationNetworks/$net_name
rm -r $broken

cd ~/optimal_recovery_sequencing

python3 find_sequence.py -n $net_name -b $broken -l 1 -r $reps --mc True

end=$(($reps-1))

for (( i=0 ; i<=$end ; i++ ));
do
python3 find_sequence.py -n $net_name -b $broken -l 1 -r $reps --mc True --damaged saved/TransportationNetworks/$net_name/$broken/$i
done

cd saved/TransportationNetworks/$net_name/$broken
cat */results.csv > combinedResults.csv

cd ~/optimal_recovery_sequencing/saved/TransportationNetworks/$net_name
mv $broken/combinedResults.csv  multiclassResults$broken.csv

cd ~/optimal_recovery_sequencing
