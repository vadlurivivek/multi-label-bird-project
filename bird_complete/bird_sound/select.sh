#!/bin/bash

for f in *; do 
	if [ -d $f ]
	then 
		echo -n $f
		ls $f | wc -l
		echo
	fi
done | sort -n -k 2 