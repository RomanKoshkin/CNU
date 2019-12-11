#!/bin/bash

##sbatch bt_serial.sh 6.5

rm timing.txt
for Ca_density in 0.01 0.051 0.092 0.134 0.175 0.216 0.258 0.299 0.34 0.381 0.423 0.464 0.505 0.546 0.588 0.629 0.67 0.711 0.753 0.794 0.835 0.876 0.918 0.959 1.0
do
    echo $Ca_density
    sleep 2
    sbatch bt_serial.sh $Ca_density
done
