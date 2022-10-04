#!/bin/bash
net_name="Anaheim"
echo $net_name
broken=8
reps=10

cd saved/TransportationNetworks/$net_name
rm -r $broken

cd ~/optimal_recovery_sequencing

python3 find_sequence.py -n $net_name -b $broken -l 1 -r $reps -d 1 2 3 4 -a True --opt True --sa True

end=$(($reps-1))

for (( i=0 ; i<=$end ; i++ ));
do
python3 find_sequence.py -n $net_name -b $broken -l 1 -r 1 -d 2 -a True --sa True --damaged saved/TransportationNetworks/$net_name/$broken/$i
done

for (( i=0 ; i<=$end ; i++ ));
do
python3 find_sequence.py -n $net_name -b $broken -l 1 -r 1 -d 3 -a True --sa True --damaged saved/TransportationNetworks/$net_name/$broken/$i
done

for (( i=0 ; i<=$end ; i++ ));
do
python3 find_sequence.py -n $net_name -b $broken -l 1 -r 1 -d 4 -a True --sa True --damaged saved/TransportationNetworks/$net_name/$broken/$i
done

cd saved/TransportationNetworks/$net_name/$broken
cat */results.csv > combinedResults.csv

cd ~/optimal_recovery_sequencing/saved/TransportationNetworks/$net_name
mv $broken/combinedResults.csv  1234Results$broken.csv

cd ~/optimal_recovery_sequencing
