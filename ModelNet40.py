import torch
import torch.utils.data

import numpy as np

import os
import os.path as osp

class ModelNet40(torch.utils.data.Dataset):

	def __init__(self, dataset_root_path, test=False):

		self.test = test

		# Build path list
		self.input_pairs = self.create_input_list(
			dataset_root_path, test)
		
	def __len__(self):
		return len(self.input_pairs)

	def __getitem__(self, idx):
		path = self.input_pairs
		# Parse the vertices, labels from the file
		vertices = self.off_vertex_parser(path)
		# Parse labels from the file
		labels = self.off_labels_parser(path)
		if not self.test:
			vertices = self.augment_data(vertices)
		# Convert numpy format to torch variable
		return [torch.from_numpy(vertices[:, idx:idx+2]), labels[idx:idx+2]]

	def create_input_list(self, dataset_root_path, test):
		input_pairs = []
		#  List of tuples grouping a label with a class
		gt_key = os.listdir(dataset_root_path)
		for obj in gt_key:
			input_pairs = dataset_root_path + '/' + obj
		return input_pairs

	def augment_data(self, vertices):
		# Random rotation about the Y-axis
		theta = 2 * np.pi * np.random.rand(1)[0]
		Ry = np.array([[np.cos(theta), 0, np.sin(theta)],
				[0, 1, 0],
				[-np.sin(theta), 0, np.cos(theta)]])
		vertices = vertices.astype(float)
		vertices = np.matmul(Ry, vertices)

		# Add Gaussian noise with standard deviation of 0.2
		vertices += np.random.normal(scale=0.02, size=(vertices.shape))
		return vertices

	def off_vertex_parser(self, path_to_off_file):
		# Read the OFF file
		with open(path_to_off_file, 'r') as f:
			contents = f.readlines()

		# Find the number of vertices contained
		# (Handle mangled header lines in .off files)
		if contents[0].strip().lower() != 'off':
			num_vertices = int(contents[0].strip()[4:].split(' ')[0])
			start_line = 1
		else:
			num_vertices = int(contents[1].strip().split(' ')[0])
			start_line = 2
		# Convert all the vertex lines to a list of lists
		vertex_list = [contents[i].strip().split(' ')[:3] for i in range(start_line, start_line+num_vertices)]
		# Return the vertices as a 3 x N numpy array
		return np.array(vertex_list).astype(float).transpose(1, 0)
	
	def off_labels_parser(self, path_to_off_file):
		# Read the OFF file
		with open(path_to_off_file, 'r') as f:
			contents = f.readlines()
		# Find the number of vertices contained
		# (Handle mangled header lines in .off files)
		if contents[0].strip().lower() != 'off':
			num_labels = int(contents[0].strip()[4:].split(' ')[0])
			start_line = 1
		else:
			num_labels = int(contents[1].strip().split(' ')[0])
			start_line = 2
		labels_list = [contents[i].strip().split(' ')[-1]
					for i in range(start_line, start_line+num_labels)]
		label_map = {"mound": 0, "ground": 1}
		labels2 = [label_map[label] for label in labels_list]
		labels_tensor = torch.tensor(labels2)
		# Return the labels as a 1 x N list
		return labels_tensor
