import numpy as np
import pickle
import os

cl = 0.
sum = 0
f_handle = open('./raw_data_all/train_no_overlap.dat', 'ab')
for root, dirs, files in os.walk("./raw_data_all/train_no_overlap", topdown=False):
		for name in dirs:
			parts = []
			parts += [each for each in os.listdir(os.path.join(root,name)) if each.endswith('.raw')]
			print(name, "...")
			for part in parts:
				example = np.loadtxt(os.path.join(root,name,part))
				i = 0
				rows = example.shape[0]
				while i <= (rows - 15):
					context = example[i:i+15,:].ravel()
					ex = np.append(context,cl)
					ex = np.reshape(ex,(1,ex.shape[0]))
					np.savetxt(f_handle, ex)
					sum += 1
					# print ex.shape
					i += 1
				print("No. of context windows: %d" % i)
			cl += 1
print("No. of Training examples: %d" % sum)
f_handle.close()

A = np.loadtxt('./raw_data_all/train_no_overlap.dat')
np.random.shuffle(A)
np.savetxt('./raw_data_all/train_no_overlap_shuf.dat',A,delimiter = ' ')
