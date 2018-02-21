#!/bin/bash

files=`cat log.txt | awk '{print $1, $3}' | sort -n -r -k 2 | cut -d' ' -f1`
for f in $files; do # iterate over bird species
	cd $f
	parts=`ls *.wav | sort -n -k1.6`
	i=0
	for c in $parts; do 	#iterate over bird species
		name="part_$i.wav"
		mv $c $name
		(( i++ ))
	done
	cd ..
done

