#!/bin/bash

files=`cat log.txt | awk '{print $1, $3}' | sort -n -r -k 2 | cut -d' ' -f1`
i=0
for f in $files; do # iterate over bird species
	(( i++ ))
	echo "No. $i Bird: $f"
	read -n 1 dummy
	cd $f
	parts=`ls *.wav | sort -n -k1.6`
	for c in $parts; do 	#iterate over bird species
		# name=$(basename $c)
		echo "Now playing: $c"
		afplay $c
		read -p "Enter r:replay  x:remove  n:exit " -n 1 opt
		echo
		while : ; do
			case $opt in
				r) echo "Now playing: $c" # to replay the sample
					afplay $c 
					read -p "Enter r:replay  x:remove  n:exit " -n 1 opt 
					echo ;;
				x) rm $c 	# to delete that sample
					echo "Removed $c" 
					continue 2
					echo ;;
				n) exit ;;
				a) echo "$i $f" >> ../abnormal.txt
					echo
					continue 2 ;;
				s) cd ..
					echo
					continue 3 ;; # to skip species
				*) echo
					continue 2 ;; # to continue to next sample
			esac
		done
		echo
	done
	echo
	cd ..
done

