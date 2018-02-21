#!/bin/bash 
#PBS -j oe 
#PBS -q day
#PBS -N cuda
## Presently only one node n81 has NVIDIA cards in it
## In future when there is a down time for the cluster 
## GPU support will be added explicitly in the scheduler
#PBS -l nodes=n81.cluster.iitmandi.ac.in:ppn=20


#PBS -V

cd $PBS_O_WORKDIR
echo 
echo "Program Output begins: " 
pwd
#-------------------------------do not modify--------------
#include the your desired nvcc compiled executable name
#or it can any package that can run on GPU cards
source activate tensorflow
python3 multi_load_nn.py

#-------------------------------do not modify--------------
echo "end of the program ready for clean up"
#-------------------------------do not modify--------------
~                                                                                                                                
~                                                                                                                                
~                                                                                                                                
~                                                                                                                                
~                                                                              

