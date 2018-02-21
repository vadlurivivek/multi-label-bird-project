import numpy as np
import pickle
import os

cl = 0.
sum = 0
f_handle = open('./data/multi_train_small_mel48.dat', 'ab')
for root, dirs, files in os.walk("./data/melfilter48/multi_train_small", topdown=False):
		for name in dirs:
			parts = []
			parts += [each for each in os.listdir(os.path.join(root,name)) if each.endswith('.mfcc')]
			print(name, "...")
			cl1 = int(name[1:3])-1 #c01_c02
			cl2 = int(name[5:7])-1
			print(cl1, cl2)
			for part in parts:
				example = np.loadtxt(os.path.join(root,name,part))
				i = 0
				rows = example.shape[0]
				while i <= (rows - 15):
					context = example[i:i+15,:].ravel()
					context1 = np.append(context,cl1)
					ex = np.append(context1, cl2)
					ex = np.reshape(ex,(1,ex.shape[0]))
					np.savetxt(f_handle, ex)
					sum += 1
					# print ex.shape
					i += 1
				print("No. of context windows: %d" % i)
			cl += 1
print("No. of Training examples: %d" % sum)
f_handle.close()

A = np.loadtxt('./data/multi_train_small_mel48.dat')
np.random.shuffle(A)
np.savetxt('./data/multi_train_small_melfilter48.dat',A,delimiter = ' ')
