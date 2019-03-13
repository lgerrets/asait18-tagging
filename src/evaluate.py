from keras.models import load_model
import csv
import numpy as np
import os

import prepare_data as pp_data
from eer import compute_eer
from parameters import *

import argparse
parser = argparse.ArgumentParser(description='Evaluate the tagger model: EER is the metric imposed by the challenge.')
parser.add_argument('--model', type=str, metavar='MD',
                    help='path to the file of the model to evaluate (required)', required=True)
parser.add_argument('--data', type=str, default="data", metavar='DA',
                    help='path to the data folder (where the files %s and %s are located) (default: data)'%(file_feat,file_y))

def main(args):
	model = load_model(args.model)

	data_test = pp_data.DataLoader(args.data,partition='evaluation')

	pred_scores = np.zeros((len(data_test),n_classes))
	true_scores = np.zeros((len(data_test),n_classes))
	result_rows = []
	chunk_refs = []

	for it,[X,y] in enumerate(data_test):
		pred = model.predict(X)

		pred = pred.mean(axis=0)
		y = y[0,:]

		pred_scores[it,:] = pred
		true_scores[it,:] = y

		chunk_ref = "chunk_"+str(it)
		chunk_refs.append(chunk_ref)

		for cl in range(n_classes):
			result_rows.append([chunk_ref, ind_to_tag[cl], pred[cl]])

	eval_result_filename = "eval_result.csv"
	with open(os.path.join(args.data, eval_result_filename),"w") as csvfile:
		writer = csv.writer(csvfile)
		writer.writerows(result_rows)

	eers = np.zeros(n_classes)
	for cl in range(n_classes):
		eers[cl] = compute_eer(os.path.join(args.data, eval_result_filename), ind_to_tag[cl], dict(zip(chunk_refs, list(true_scores[:,cl]))))

	print(eers)


if __name__ == '__main__':
	args = parser.parse_args()
	main(args)
