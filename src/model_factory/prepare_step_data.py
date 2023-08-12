from numpy import array
from numpy import hstack
import pandas as pd

def prepare_step_data_return(sequence, n_steps):
	X, y = [], []
	for i in range(len(sequence)):
		end_ix = i + n_steps
		if end_ix > len(sequence)-2:
			break
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix+1]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

def prepare_mulitvariate_data_step(values, steps):
	X = list()
	for i in range(len(values)):
		# find the end of this pattern
		end_ix = i + steps
		# check if we are beyond the dataset
		if end_ix > len(values):
			break
		# gather input and output parts of the pattern
		seq_x = values[i:end_ix, :-1]
		X.append(seq_x)
	return array(X)