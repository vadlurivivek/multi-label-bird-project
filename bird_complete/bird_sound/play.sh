#!/bin/bash

files=`cat log.txt | awk '{print $1, $3}' | sort -n -r -k 2 | cut -d' ' -f1`
i=1
for f in $files; do # iterate over bird species
	echo "No. $i Bird: $f"
	read -n 1 dummy
	parts=`ls -1 $f/*.wav | sort -n -t"_" -k1.6`
	# echo $parts
	for c in $parts; do 	#iterate over bird species
		name=$(basename $c)
		echo "Now playing: $name"
		echo $c
		afplay $c
		read -p "Enter r:replay  x:remove  n:exit " -n 1 opt
		echo
		f=r
		while [ $f == r ]; do
			case $opt in
				r) echo "Now playing: $name" # to replay the sample
					afplay $c 
					read -p "Enter r:replay  x:remove  n:exit " -n 1 opt 
					echo ;;
				x) rm $c 	# to delete that sample
					echo "Removed $name" 
					f=n
					echo ;;
				n) exit ;;
				a) echo "$i $name" >> abnormal.txt
					continue 3 ;;
				s) continue 3 ;; # to skip species
				*) continue 2 ;; # to continue to next sample
			esac
		done
		echo
	done
	echo
	(( i++ ))
done

