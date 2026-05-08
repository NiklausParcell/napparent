# cython: language_level=3


import numpy as np 
import pandas as pd 
import pyarrow.parquet as pq 
import time
import sys 
from pandas.api.types import CategoricalDtype
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from itertools import combinations
from functools import wraps 
# from numba import njit

def block_print(func):
	@wraps(func)
	def wrapper(*args, **kwargs):
		# do something before the function is called
		result = func(*args, **kwargs)
		# do something after the function is called
		return result
	return wrapper

def load_numpy(df, cols_to_drop, target):

	cols_to_drop.append(target)
	# X = df.drop(cols_to_drop, axis=1)
	X = df.drop(target, axis=1)
	Y = df[target]

	# Store columns 
	X_cols = X.columns
	k = 0 
	cg = {}  # cg == column graph
	cd = {}  # cd == columns dropped 
	for col in X_cols:
		cg[k] = col
		if col in cols_to_drop:
			cd[k] = col
		k += 1

	# convert to numpy 
	X = X.to_numpy()
	Y = Y.to_numpy()

	print("Data loaded...")
	return X, Y, cg, cd 


def assign_outcome(X, Y, use_ML):
	"""
	Train
	- use XGBoost to train a model on the processed data 
	"""
	if use_ML == True:
		print("Training model...")
	print("Finding outcome...")

	# Define data type 
	# dtype = pd.DataFrame(Y).dtypes[0]
	dtype = pd.DataFrame(Y).dtypes.iloc[0]

	# Handle different model types 
	if isinstance(dtype, CategoricalDtype) or str(dtype) == "object":

		from xgboost import XGBClassifier

		return_dtype = "Categorical"

		# Need to vectorize X and Y 
		ohe = OneHotEncoder()
		vectors = ohe.fit_transform(X)

		Y = Y.fillna("nan")

		# Split processed in to X and Y 
		le = LabelEncoder()
		labels = le.fit_transform(Y)

		# *Robot voice* "initiate training sequence"
		model = XGBClassifier() 
		model.fit(vectors, labels)

		# outcomes 
		outcomes = model.predict(vectors)

		def validate():
			# Multi-Categorical validation scores
			print("Validating...")
			accuracy = np.mean(outcomes == labels)
			print("Accuracy: ", accuracy)
			return {"Accuracy": accuracy}
		# data_viability_score = validate() 

		# Inverse transform labels back to original
		outcomes = le.inverse_transform(outcomes)

		# outcomes for barnacle 
		outcomes = model.predict_proba(vectors) 
	else:
		"""
		For numerical, it is not always necessary to use a ML model... 
		"""
		return_dtype = "Numerical"

		# Y = Y.fillna(0)

		# numpy equivalent of ffill 
		Y = np.nan_to_num(Y, nan=0)

		# use_ML = False 
		if use_ML == True:
			print("Using ML to assign...")
			from xgboost import XGBRegressor
			# Need to vectorize X and Y 
			ohe = OneHotEncoder()
			vectors = ohe.fit_transform(X)
			labels = Y
			model = XGBRegressor() 
			model.fit(vectors, labels)
			outcomes = model.predict(vectors) 

			def validate():
				# Multi-Categorical validation scores
				print("Validating...")

				# Perform a R2 
				r2 = model.score(vectors, labels)
				mae = mean_absolute_error(labels, outcomes)
				print("R2: ", r2)
				print("MAE: ", mae)
				return {"R2": r2, "MAE": mae}
			# data_viability_score = validate() 
		else:
			outcomes = np.array(Y) 

	# print("Assigned modeling type is: ", return_dtype)
	return np.array(outcomes), return_dtype

class preproc_stream_numba:

	def round_to_significant_figures(self, arr, n=4):
		"""
		Round an array of numbers to n significant figures.
		
		Parameters:
		- arr: NumPy array of numbers to round.
		- n: Number of significant figures.
		
		Returns:
		- NumPy array of numbers rounded to n significant figures.
		"""
		arr = arr.astype(np.float32)

		# Avoid division by zero for zeros in the array
		nonzero = arr != 0
		# Scale numbers to have the digit of interest right before the decimal point
		scale = np.log10(np.abs(arr[nonzero])) // 1 - (n - 1)
		# Round the scaled numbers and scale back
		rounded = np.round(arr[nonzero] / (10 ** scale)) * (10 ** scale)
		# Create a result array with the same shape as the input
		result = np.zeros_like(arr)
		result[nonzero] = rounded
		# Ensure the result is of float type
		return result.astype(np.float32)

class preprocess_stream():
	def __init__(self, cols_to_drop, target):
		self.preprocess_map = {} 
		self.num_chunks = 0 
		self.cols_to_drop = cols_to_drop
		self.cols_to_drop.append(target) 
		self.target = target 

	def count_categorical(self, values):
		"""
		Count categorical values
		-> values are the pandas dataframe column: df[col] 
		"""
		counts = values.value_counts()
		vals = list(counts.index)
		hist_vals = list(counts.values)
		return hist_vals, vals 

	def round_values(self, vals):
		return ["{:.3g}".format(float(val)) for val in vals]
	
	def round_to_significant_figures(self, arr, n=4):
		"""
		Round an array of numbers to n significant figures.
		
		Parameters:
		- arr: NumPy array of numbers to round.
		- n: Number of significant figures.
		
		Returns:
		- NumPy array of numbers rounded to n significant figures.
		"""
		arr = arr.astype(np.float32)

		# Avoid division by zero for zeros in the array
		nonzero = arr != 0
		# Scale numbers to have the digit of interest right before the decimal point
		scale = np.log10(np.abs(arr[nonzero])) // 1 - (n - 1)
		# Round the scaled numbers and scale back
		rounded = np.round(arr[nonzero] / (10 ** scale)) * (10 ** scale)
		# Create a result array with the same shape as the input
		result = np.zeros_like(arr)
		result[nonzero] = rounded
		# Ensure the result is of float type
		return result.astype(np.float32)

	def _initialize_preprocess_map(self, X, cg, cd):
		"""
		Initializes preprocess map
		"""
		self.col_graph = cg 
		self.cols_dropped = cd
		self.cols = [k for k in self.col_graph if k not in self.cols_dropped]
		self.col_names = [self.col_graph[k] for k in self.col_graph]
		self.preprocess_map = {}
		for col in self.cols:
			arr = X[:, col]
			data_check = str(type(arr[0]))
			# if arr.dtype == 'float64' or arr.dtype == 'int64':
			if "int" in data_check or "float" in data_check:
				# vals = self.round_values(arr) 
				arr = self.round_to_significant_figures(arr)
				# arr = np.nan_to_num(arr, nan=0)
				arr = arr.astype(np.float32) 
				arr = np.nan_to_num(arr, nan=0)
				
				bin_type = "numerical" 
				
				min_val = arr.min()
				max_val = arr.max()
				if col == 14:
					debug=1
				self.preprocess_map[col] = {} 
				self.preprocess_map[col]["min"] = min_val
				self.preprocess_map[col]["max"] = max_val

				
			else:
				arr = np.nan_to_num(arr, nan="empty")
				bin_type = "categorical"
				self.preprocess_map[col] = {}
			
			unique_values, counts = np.unique(arr, return_counts=True)
			temp = dict(zip(unique_values, counts))
			self.preprocess_map[col]["values"] = {} 
			for val in temp:
				if "nan" not in str(val):
					if val in self.preprocess_map[col]:
						self.preprocess_map[col]["values"][val] += temp[val]
					else:
						self.preprocess_map[col]["values"][val] = temp[val]
				else:
					if "nan" in self.preprocess_map[col]["values"]:
						self.preprocess_map[col]["values"]["nan"] += temp[val]
					else:
						self.preprocess_map[col]["values"]["nan"] = temp[val]
			# self.preprocess_map[col]["values"] = temp
			self.preprocess_map[col]["bin_type"] = bin_type

		self.num_chunks += 1
	
	def _update_preprocess_map(self, X):
		"""
		Updates preprocess map
		"""
		for col in self.cols:
			arr = X[:, col]
			if self.preprocess_map[col]["bin_type"] == "numerical":
				# vals = self.round_values(arr)
				arr = self.round_to_significant_figures(arr)
				arr = arr.astype(np.float32) 
				arr = np.nan_to_num(arr, nan=0)

				min_val = arr.min()
				max_val = arr.max()
				if col == 14:
					debug=1
				if min_val < self.preprocess_map[col]["min"]:
					self.preprocess_map[col]["min"] = min_val
				if max_val > self.preprocess_map[col]["max"]:
					self.preprocess_map[col]["max"] = max_val

				
			else:
				arr = np.nan_to_num(arr, nan="empty")

			unique_values, counts = np.unique(arr, return_counts=True)
			temp = dict(zip(unique_values, counts))
			for val in temp:
				if "nan" not in str(val):
					if val in self.preprocess_map[col]["values"]:
						self.preprocess_map[col]["values"][val] += temp[val]
					else:
						self.preprocess_map[col]["values"][val] = temp[val]
					debug=1
				else:
					if "nan" in self.preprocess_map[col]["values"]:
						self.preprocess_map[col]["values"]["nan"] += temp[val]
					else:
						self.preprocess_map[col]["values"]["nan"] = temp[val]

		self.num_chunks += 1

	def preprocess(self, X, cg, cd):
		"""
		Preprocess chunk
		If self.num_chunks = 0, initialize preprocess map, otherwise work with preprocess map 
		"""
		if self.num_chunks == 0:
			self._initialize_preprocess_map(X, cg, cd)
		else:
			self._update_preprocess_map(X)

	def finish_map(self, barnacle_depth):
		"""
		Finish preprocessing
		"""
		# Bin edges
		for col in self.preprocess_map:
			if col in barnacle_depth:
				barn_depth = barnacle_depth[col]
			else:
				barn_depth = barnacle_depth["main"] 

			# Drop "nan" from values in the values dict
			if "nan" in self.preprocess_map[col]["values"]:
				del self.preprocess_map[col]["values"]["nan"]
				
			
			# bins = sorted(self.preprocess_map[col]["values"].items(), key=lambda x: x[1], reverse=True)[:barn_depth-1] 
			bins = self.preprocess_map[col]["values"].items()

			def return_accounted_bins(bins, barn_depth):
				bins = list(bins) 
				new_bins = sorted(bins, key=lambda x: x[1], reverse=True)
				arr = np.array(new_bins) 
				new_bins_barn = arr[:barn_depth]

				if barn_depth > len(bins):
					return new_bins_barn
				else:
					iterater = barn_depth 
				done_adding = 0
				# barn_last = bins[barn_depth-1][1]  # the alleged last in the sequence at barn_depth
				barn_last = new_bins[barn_depth-1][1]  # the alleged last in the sequence at barn_depth
				while done_adding == 0 and iterater < len(new_bins):
					# print(iterater)
					if new_bins[iterater][1] == barn_last:
						new_bins_barn = arr[:iterater]
						iterater += 1
					else:
						done_adding = 1

				debug=1
				return new_bins_barn
			
			bins = return_accounted_bins(bins, barn_depth)
			bins = dict(bins)

			# Sort bins to a list with no duplicates 
			bins_sorted = list(set(bins.keys()))
			min_val = self.preprocess_map[col]["min"]
			max_val = self.preprocess_map[col]["max"]
			if min_val < min(bins_sorted):
				bins_sorted = [min_val] + bins_sorted
			if max_val > max(bins_sorted):
				# bins_sorted = bins_sorted + [max_val]
				bins_sorted[-1] = max_val
			bins_sorted = sorted([np.float32(val) for val in bins_sorted])

			# Compile list of bins and bin_range with bottom, top 
			bin_graph = {} 
			for k, bin_val in enumerate(bins_sorted):
				if k < len(bins_sorted) - 1:
					bin_graph[bin_val] = {
						"bin": bin_val,
						"bin_range": {
							"bottom": bin_val,
							"top": bins_sorted[k+1]
						}
					}

			# Precompile bin_labels 
			bin_labels = [] 
			
			if self.preprocess_map[col]["bin_type"] == "numerical":
				for k, bin_val in enumerate(bin_graph):
					top = bin_graph[bin_val]["bin_range"]["top"]
					bottom = bin_graph[bin_val]["bin_range"]["bottom"]

					if abs(bottom) < 10000 and abs(bottom) > 0.1:
						bin_1 = f"{round(bin_val, 4)}"
					else:
						bin_1 = f"{bin_val:.4g}"
					
					if abs(top) < 10000 and abs(top) > 0.1:
						bin_2 = f"{round(bin_graph[bin_val]['bin_range']['top'], 4)}"
					else:
						bin_2 = f"{top:.4g}"

					bin_label = bin_1 + " - " + bin_2
					bin_labels.append(bin_label)
					bin_graph[bin_val]["bin_label"] = bin_label
			else:
				# bin_labels = list(set(vals)) 
				bin_labels = list(bins.keys())
			
			# Sort bin_labels
			# bin_labels = sorted(bin_labels)	
			if col == 14:
				debug=1
			self.preprocess_map[col]["bins"] = bins
			self.preprocess_map[col]["bin_labels"] = bin_labels 
			self.preprocess_map[col]["bin_graph"] = bin_graph

		print("Preprocess map finished...")

	
	def map_numerical_bins(self, col, val):
		"""
		Map to bins
		"""
		bin_graph = self.preprocess_map[col]["bin_graph"]
		# bin_graph = np.array(list(bin_graph.items()))
		for bin_val in bin_graph:
			top = bin_graph[bin_val]["bin_range"]["top"]
			bottom = bin_graph[bin_val]["bin_range"]["bottom"]
			if val >= bottom and val < top:
				bin_label = bin_graph[bin_val]["bin_label"]
				return bin_label
		return "other"

	def vectorized_map_numerical_bins(self, col, arr):
		# Preprocess a lil bit 
		arr = arr.astype(np.float32) 
		arr[np.isnan(arr)] = 0 

		# Bin
		bin_graph = self.preprocess_map[col]["bin_graph"]
		bin_list = [bin_graph[bin_val]["bin_range"]["bottom"] for bin_val in sorted(bin_graph)]
		bin_edges = np.array(bin_list)
		# bin_edges = np.array([bin_graph[bin_val]["bin_range"]["bottom"] for bin_val in sorted(bin_graph)])
		bin_labels = np.array([bin_graph[bin_val]["bin_label"] for bin_val in sorted(bin_graph)])
		# bin_labels = bin_labels + ["other"]
		bin_indices = np.digitize(arr, bin_edges, right=False) - 1
		return_bins = bin_labels[bin_indices]
		# if self.col_graph[col] == "census_age_median":
		# 	N = len(arr) 
		# 	self.preprocess_map[col]["error"] = arr[N-1000:N]
		# 	debug=1
		return return_bins

			
	def map_categorical_bins(self, col, val):
		"""
		Map to bins
		"""
		bins_map = self.preprocess_map[col]["values"]
		if val in bins_map:
			return val
		return "other"
	

	def use_map(self, chunk):
		"""
		chunk is a chunk of a numpy array now 
		"""
		print("Using preprocess network...")
		# Map to bins
		processed = {}
		# N = len(self.cols)
		N = len(self.col_graph)
		# for k in self.cols:
		for k in self.col_graph:
			col = self.col_graph[k]
			arr = chunk[:, k]
			# print(f"Processing player {k+1} of {N}                  ", end="\r")
			if k in self.preprocess_map: 
				if self.preprocess_map[k]["bin_type"] == "numerical":
					# processed[col] = np.array([self.map_numerical_bins(k, val) for val in arr])
					arr = np.nan_to_num(arr, nan=0)
					processed[col] = self.vectorized_map_numerical_bins(k, arr)
				else:
					arr = np.nan_to_num(arr, nan="empty")
					processed[col] = np.array([self.map_categorical_bins(k, val) for val in arr])

			else:
				processed[col] = arr

		# processed = pd.DataFrame(processed)
		return processed   # dict of numpy arrays 
	

"""
PB classes 
"""
def help_X_col(X_j):
	"""
	X_j is X[col]
	"""
	data_type = str(type(X_j[0]))
	if "int" in data_type or "float" in data_type:
		is_not_number = np.vectorize(lambda x: not isinstance(x, (int, float)))
		replaced = np.where(is_not_number(X_j), 0, X_j).astype(np.float32)
	else:
		is_not_string = np.vectorize(lambda x: not isinstance(x, str))
		replaced = np.where(is_not_string(X_j), "no data", X_j)
	
	# Replace None with "None"
	is_none = np.vectorize(lambda x: x is None)
	replaced = np.where(is_none(replaced), "None", replaced)

	return replaced


class PsychicBarnacle():

	def __init__(self):
		self.wegoing = "yes" 
	
	
	def initialize_inputs(self, first_X_input, first_Y_input, first_outcomes_input, return_dtype, col_info, target):
		"""
		This class is meant to extend the framework of nnm from Psychic Barnacle to a streaming context. Streaming is used for large datasets > 100MB that can't be processed individually, so have to be processed in batches. 

		Parameters:
			first_X_input: pandas dataframe
			first_Y_input: pandas series
			first_outcomes_input: pandas dataframe
			return_dtype: str, "Categorical" or "Numerical"
		"""
		
		# Define target and outcomes 
		self.target = target
		self.outcomes_input = np.array(first_outcomes_input)

		# IMPORTANT: num_chunks, probably used everywhere 
		self.num_chunks = 0

		# Define components to be used later 
		self.outcomes = self.outcomes_input

		# For averages 
		self.avg_outcome = 0
		self.avg_count = 0 

		"""
		Create initial col and val maps (from str <> int)
		"""
		self.k = 0 
		self.col_map_int, self.col_map_str = {}, {}
		self.val_map_int, self.val_map_str = {}, {}

		# Define from col_info graph 
		self.col_graph = col_info["col_graph"]
		self.cols_dropped = col_info["cols_dropped"]
		self.cols = [k for k in self.col_graph if k not in self.cols_dropped]
		self.col_names = [self.col_graph[k] for k in self.col_graph]

		# Set up column map before values map 
		# for col in first_X_input.columns:
		for col in first_X_input:
			col = str(col)
			if col not in self.col_map_str:
				self.col_map_int[self.k], self.col_map_str[col] = col, self.k
				self.k += 1

	def make_col_combos(self, X):
		"""
		Define column combinations
		"""
			
		if self.num_chunks == 0:
			print("Setting up player map...")
			self.vals_map = {} 

			print("Setting up player combinations...")
			# self.col_combos = combinations(X.columns, 2)
			self.col_combos = combinations(self.cols, 2) 
			self.col_combos = list(self.col_combos)
			self.temp_col_combos, self.tup_combos = {}, {} 
			for k, combo in enumerate(self.col_combos):
				self.temp_col_combos[k] = tuple(combo)
				self.tup_combos[k] = (combo[0], combo[1])
			self.col_combos = self.temp_col_combos
			self.col_array = np.array(list(self.col_combos.keys()))


			print("Setting up player id map...")
			self.col_to_tup = {}
			for c in self.tup_combos:
				tup_combo = self.tup_combos[c]
				col1, col2 = tup_combo[0], tup_combo[1]
				self.col_to_tup[col1] = [] 
				self.col_to_tup[col2] = []
				self.col_to_tup[col1].append(tuple(sorted(self.tup_combos[c])))
				self.col_to_tup[col2].append(tuple(sorted(self.tup_combos[c])))

			self.col_to_tup = {col: tuple(self.col_to_tup[col]) for col in self.col_to_tup}

	# def help_X_col(self, X_j, col):
	# 	"""
	# 	X_j is X[col]
	# 	"""
	# 	data_type = str(type(X_j[0]))
	# 	if "int" in data_type or "float" in data_type:
	# 		is_not_number = np.vectorize(lambda x: not isinstance(x, (int, float)))
	# 		replaced = np.where(is_not_number(X_j), 0, X_j).astype(np.float32)
	# 	else:
	# 		is_not_string = np.vectorize(lambda x: not isinstance(x, str))
	# 		replaced = np.where(is_not_string(X_j), "no data", X_j)
		
	# 	# Replace None with "None"
	# 	is_none = np.vectorize(lambda x: x is None)
	# 	replaced = np.where(is_none(replaced), "None", replaced)

	# 	return replaced
	
	def convert_ff_to_array(self, X):
		N = len(X[self.col_names[0]])
		M = len(self.col_graph)
		temp = np.empty((N, M), dtype = object)
		for col in X:
			if col in self.cols:
				temp[:, self.col_map_str[col]] = self.help_X_col(X[col], col)
			else:
				temp[:, self.col_map_str[col]] = X[col]
			debug=1
		return temp 
	
	def X_to_vec(self, X):
		"""
		"""
		# 1) Create X array of processed only
		N = len(X[:, 0])
		M = len(self.cols) 
		X_proc = np.empty((N, M), dtype = object)
		for k, col in enumerate(self.cols):
			X_proc[:, k] = X[:, col]
		
		# 2) Vectorize X
		val_str = np.vectorize(self.map_val_str)
		X_mapped = val_str(X_proc)

		# 3) Create fast-frame
		X_mapped_ff = {}
		for k, col in enumerate(self.cols):
			X_mapped_ff[col] = X_mapped[:, k].astype(np.int32)
		for col in self.cols_dropped:
			X_mapped_ff[col] = X[:, col]

		return X_mapped_ff
	
	def X_to_mapped(self, X):
		"""
		"""
		print("Stacking players...")
		# 1) Create X array of processed only
		N = len(X[:, 0])
		M = len(self.cols) 
		X_proc = np.empty((N, M), dtype = object)
		for k, col in enumerate(self.cols):
			X_proc[:, k] = X[:, col]
		
		# 2) Vectorize X
		val_str = np.vectorize(self.map_val_str)
		X_mapped = val_str(X_proc)

		# 3) Make X_mapped the shape of the original X 
		X_mapped_ff = np.empty((N, len(self.col_graph)), dtype = object)
		for k, col in enumerate(self.cols):
			X_mapped_ff[:, col] = X_mapped[:, k].astype(np.int32)
		for col in self.cols_dropped:
			X_mapped_ff[:, col] = X[:, col]

		# 3) For combo in self.tup_combos, np.vstack and add to frame 
		X_mapped_vec = np.empty((N, len(self.col_array), 2), dtype = np.int32)
		for c in self.col_array:
			combo = self.tup_combos[c]
			X_mapped_vec[:, c] = np.vstack((X_mapped_ff[:, combo[0]],  X_mapped_ff[:, combo[1]])).T.astype(np.int32) 

		return X_mapped_vec

	def val_checking(self, X):
		"""
		Initializing and/or updating of:
		- col_map_int, col_map_str
		- val_map_int, val_map_str
		"""
		print("Checking player positions...")

		"""
		TODO: Vectorize this
		"""
		if self.num_chunks == 0:
			if "no data" not in self.val_map_str:
				self.val_map_int[self.k], self.val_map_str["no data"] = "no data", self.k
				self.k += 1
			if "None" not in self.val_map_str:
				self.val_map_int[self.k], self.val_map_str["None"] = "None", self.k
				self.k += 1

		for col in self.cols:
			# X[col] = self.help_X_col(X[col])
			# for val in X[col].unique():
			for val in np.unique(X[:, col]):
				val = str(val)
				if val not in self.val_map_str:
					# save_val = f"{col}_{val}"
					self.val_map_int[self.k], self.val_map_str[val] = val, self.k
					self.k += 1


	def map_val_str(self, val):
		return self.val_map_str[val]
	

	def vals_map_updating(self, X, outcomes):
		"""
		1) Preallocate to X_mapped
		2) Update vals_map
		"""

		# print("Preallocating player values...")
		print(f"Using player map on chunk {self.num_chunks}...")


		# New fast-frame to array 
		X = self.convert_ff_to_array(X)
		self.val_checking(X) 
		X_mapped = self.X_to_vec(X)
		# X_mapped = self.X_to_mapped(X)


		outs = np.array(outcomes).astype(np.float32)
		self.avg_outcome = outs.sum() 
		self.avg_count += len(outs)

		# One percent number for significance 
		one_percent = int(len(X_mapped[self.col_array[0]]) * 0.01)

		# print(f"Checking player outcomes, with 1% as {one_percent}...")
		print("Checking player outcomes...")
		for c in self.col_array:
			# if c % print_every == 0:
			# 	perc = int(c * 100 / len_col_combos)
			# 	print(f"{perc}% complete...", end="\r")
			combo = self.tup_combos[c]

			# stack_time = time.time()
			vals = np.vstack((X_mapped[combo[0]], X_mapped[combo[1]]))
			vals = vals.T
			# print("Stack time: ", time.time() - stack_time)

			# uniques = set(vals)
			uniques, counts = np.unique(vals, axis=0, return_counts=True)
			uniques_ = uniques[counts > one_percent]
			uniques_not_in = uniques[counts <= one_percent]
			# for k, unique_arr in enumerate(uniques):
			for unique_arr in uniques_:

				# tup_creation = time.time()
				unique_inv_arr = unique_arr[::-1]
				unique_tup, unique_inv_tup = tuple(unique_arr), tuple(unique_inv_arr)
				# print("Tup creation time: ", time.time() - tup_creation)

				# index_time = time.time()
				indices = np.where((vals == unique_arr).all(axis=1))[0].astype(np.int32)
				indices_inv = np.where((vals == unique_inv_arr).all(axis=1))[0].astype(np.int32)
				# print("Index time: ", time.time() - index_time)

				# N_time = time.time()
				N = np.array(len(indices)).astype(np.float32)
				N_inv = np.array(len(indices_inv)).astype(np.float32)
				# print("N time: ", time.time() - N_time)
				
				# filter_time = time.time()
				outs_filtered = outs[indices]
				outs_filtered_inv = outs[indices_inv]
				# print("Filter time: ", time.time() - filter_time)
				
				# sum_time = time.time()
				outcome = np.sum(outs_filtered).astype(np.float32)
				outcome_inv = np.sum(outs_filtered_inv).astype(np.float32)
				# print("Sum time: ", time.time() - sum_time)
				
				# if_time = time.time()
				if unique_tup not in self.vals_map:
					self.vals_map[unique_tup] = np.array([0, 0]).astype(np.float32)
				if unique_inv_tup not in self.vals_map:
					self.vals_map[unique_inv_tup] = np.array([0, 0]).astype(np.float32)
				# print("If time: ", time.time() - if_time)
				
				# vals_time = time.time()
				self.vals_map[unique_tup][0] += outcome
				self.vals_map[unique_inv_tup][0] += outcome_inv
				self.vals_map[unique_tup][1] += N
				self.vals_map[unique_inv_tup][1] += N_inv
				# print("Vals time: ", time.time() - vals_time)
				# sys.exit() 

			for unique_arr in uniques_not_in:
				unique_inv_arr = unique_arr[::-1]
				unique_tup, unique_inv_tup = tuple(unique_arr), tuple(unique_inv_arr)
				if unique_tup not in self.vals_map:
					self.vals_map[unique_tup] = np.array([0, 0]).astype(np.float32)
				if unique_inv_tup not in self.vals_map:
					self.vals_map[unique_inv_tup] = np.array([0, 0]).astype(np.float32)
				self.vals_map[unique_tup][0] += 0
				self.vals_map[unique_inv_tup][0] += 0
				self.vals_map[unique_tup][1] += 0
				self.vals_map[unique_inv_tup][1] += 0

		self.num_chunks += 1
		return 
	

	def finish_map(self):
		print("Finishing player map...")
		self.vals_map_avg = {} 
		unacceptables = [0, 1]
		for val_tup in self.vals_map:
			v0, v1 = val_tup[0], val_tup[1]

			# Outcome0
			count0 = self.vals_map[val_tup][1]
			if count0 in unacceptables:
				div0 = 0
			else:
				outcome0 = self.vals_map[val_tup][0]
				multiplier0 = np.log10(count0)  # accounts for frequency of val_tup 
				div0 = (outcome0 / count0) * multiplier0

			self.vals_map_avg[val_tup] = div0
			
			# Outcome1
			val_tup_inv = (v1, v0)
			count1 = self.vals_map[val_tup_inv][1]
			if count1 in unacceptables:
				div1 = 0
			else:
				outcome1 = self.vals_map[val_tup_inv][0]
				multiplier1 = np.log10(count1)
				div1 = (outcome1 / count1) * multiplier1

			self.vals_map_avg[val_tup_inv] = div1

		self.avg_outcome = self.avg_outcome / self.avg_count

	
	def make_cvto_from_existing(self, X):
		"""
		Preallocate column values
		"""
		print("Player-Player cvt from outcomes...")
		# New fast-frame to array 
		X = self.convert_ff_to_array(X)
		self.val_checking(X) 
		X_mapped = self.X_to_vec(X)
		
		M = len(self.col_array)
		col_vals_tup = np.empty((len(X), M), dtype = object)
		print_every = M // 10
		for c in self.col_array:
			# print("Interaction combo %s of %s"%(c, M), end="\r")
			# if c % print_every == 0:
			# 	perc = int(c * 100 / M)
			# 	print(f"{perc}% complete...", end="\r")
			combo = self.col_combos[c] 
			col1, col2 = combo[0], combo[1]
			# col1_vals, col2_vals = X_mapped[col1].astype(np.int32), X_mapped[col2].astype(np.int32)
			col1_vals, col2_vals = X_mapped[col1], X_mapped[col2]
			vals = tuple(zip(col1_vals, col2_vals))
			col_vals_tup[:, c] = vals

		return col_vals_tup
	

	def map_avg_with_vec(self, val_tup):
		return self.vals_map_avg[val_tup]
	

	def map_to_avg_vec(self, X): 
		print("Player-Player mapping to outcomes...")
		val_str = np.vectorize(self.map_avg_with_vec)
		col_vals_outcomes = val_str(X)
		return col_vals_outcomes

	def use_map(self, X, og_X, Y, outcomes):
		print("Using deep network...")
		outcomes = np.array(outcomes) 
		col_vals_tup = self.make_cvto_from_existing(X)
		col_vals_outcomes = self.map_to_avg_vec(col_vals_tup)
		

		# Combine columns by averaging column combination by column
		print("Looking at players in the game together...")
		# N = len(col_vals_outcomes)
		# for c in col_combined:
		# 	col_combined[c] = np.zeros(N)
		col_combined = {} 
		M = len(self.col_to_tup)-1
		for c in self.tup_combos:
			combo = self.tup_combos[c]
			c1, c2 = combo[0], combo[1]
			if c1 not in col_combined:
				col_combined[c1] = col_vals_outcomes[:, c]
			else:
				col_combined[c1] += col_vals_outcomes[:, c]
			if c2 not in col_combined:
				col_combined[c2] = col_vals_outcomes[:, c]
			else:
				col_combined[c2] += col_vals_outcomes[:, c]
		for c in col_combined:
				col_combined[c] = col_combined[c] / M

		nnm_diffs = {}
		for c in self.col_graph:
			col = self.col_graph[c]
			nnm_diffs[col] = X[col]


		# combine 
		print("Analyzing players and outcomes...")
		for c in col_combined:
			col = self.col_map_int[c]
			# nnm_diffs[f"{col}_pst"] = np.array(X[col])
			col_combined_c_g = col_combined[c]
			diffs = col_combined_c_g - self.avg_outcome[g]
			# diffs = col_combined_c_g
			# diffs = col_combined_c_g - outcomes 
			nnm_diffs[col + "_barn"] = diffs

		nnm_diffs["Actuals"] = np.array(Y)
		# nnm_diffs["outcomes_" + self.target] = outcomes 
		nnm_diffs["outcomes_barn"] = outcomes

		nnm_diffs = pd.DataFrame(nnm_diffs)
		return nnm_diffs 
	

def test_preprocess():
	path = "./data/TN_Jerry4.csv"
	chunk_sizes = [
		1000,
		1500,
		# 100000,
		# 130000
	]
	df = pd.read_csv(path)
	# df = df[:10000]
	df = df[:3000]

	cols_to_drop = [
		"npi",
		"billing_code",
		"npi_zip5",
		"census_zip",
		"census_city",
		"census_lat",
		"census_lng",
		"census_county_fips",
		"census_county_name",
		"median",
		# "std",
		"count",
		"family",
		"sub_family",
		# "npi_count",
	]

	barnacle_depth = {
		"main": 8,
	}

	target = "std" 

	# Define X, X_untouched, Y, col_graph 
	X, Y, col_graph, c_dropped = load_numpy(df, cols_to_drop, target)

	pm = {} 
	pdd = {} 
	for chunk_size in chunk_sizes:

		# Total time 
		total_start_time = time.time()

		# Preprocess
		start_time = time.time()
		pst = preprocess_stream(cols_to_drop=cols_to_drop, target="std")
		num_chunks = 0
		for i in range(0, len(df), chunk_size):
			chunk = X[i:i+chunk_size]
			# pst.preprocess(chunk)
			pst.preprocess(chunk, col_graph, c_dropped)
			num_chunks += 1
			print(f"Chunk {num_chunks} processed")
		print(f"Preprocessing time for chunk size {chunk_size}: ", time.time() - start_time)
		pst.finish_map(barnacle_depth)

		num_chunks = 0
		for i in range(0, len(df), chunk_size):
			use_time = time.time()
			chunk = X[i:i+chunk_size]
			processed = pst.use_map(chunk)
			if num_chunks == 0:
				process_map = processed
			else:
				# Add to processed
				process_map = pd.concat([process_map, processed])
			num_chunks += 1
			# print(f"Chunk {num_chunks} processed")
			print(f"{chunk_size} for chunk {num_chunks} processed in : ", time.time() - use_time)

		# Total time
		print(f"Total time for chunk size {chunk_size}: ", time.time() - total_start_time)

		# Save processed
		process_map.to_csv(f"./data/processed_{chunk_size}.csv")

		pdd[chunk_size] = process_map
		pm[chunk_size] = pst.preprocess_map

	# Check column values are the same
	print("\n\n\n Checking if preprocess maps are matching")
	checks = {
		col: pm[chunk_sizes[0]][col] == pm[chunk_sizes[1]][col] for col in pm[chunk_sizes[0]]
	}
	for col in checks:
		if checks[col] == False:
			print(col, checks[col])

	# Check column values are the same 
	print("\n\n\n Checking column matching") 
	checks = {
		col: list(pdd[chunk_sizes[0]][col]) == list(pdd[chunk_sizes[1]][col]) for col in list(pdd[chunk_sizes[0]].columns)
	}
	for col in checks:
		if checks[col] == False:
			print(col, checks[col])
			p1 = pdd[chunk_sizes[0]][col]
			p2 = pdd[chunk_sizes[1]][col]
			p1 = np.array(p1) 
			p2 = np.array(p2)
			wheres = np.where(p1 != p2)
			p1_diff = p1[wheres]
			p2_diff = p2[wheres]
			print(np.unique(p1))
			print(np.unique(p2))
			print(p1)
			print(p2)
			debug=1
	debug=1

	

def pb_stream_algorithm():

	path = "./data/TN_Jerry4.csv"
	df = pd.read_csv(path)
	test = "full"
	if test == "full":
		chunk_sizes = [
			len(df) // 5,
			len(df) // 4,
		]
	else:
		chunk_sizes = [
			1000, 
			1500
		]
		df = df[:3000]  # Testing 
	target = "std" 

	cols_to_drop = [
		"npi",
		"billing_code",
		"city",
		"npi_zip5",
		"census_zip",
		"census_city",
		"census_lat",
		"census_lng",
		"census_county_fips",
		"census_county_name",
		"median",
		# "std",
		"count",
		"family",
		"sub_family",
		# "npi_count",
	]

	barnacle_depth = {
		"main": 8,
	}

	# Define X, X_untouched, Y, col_graph 
	X, Y, col_graph, c_dropped = load_numpy(df, cols_to_drop, target)


	val_maps = {} 
	check = {} 
	for chunk_size in chunk_sizes:
		if chunk_size > len(df):
			print("Chunk size is larger than dataframe, skipping")
			sys.exit() 

		# Preprocess
		start_time = time.time()
		pst = preprocess_stream(cols_to_drop=cols_to_drop, target=target)
		num_chunks = 0
		for i in range(0, len(X), chunk_size):
			chunk = X[i:i+chunk_size]
			# pst.preprocess(chunk)
			pst.preprocess(chunk, col_graph, c_dropped)
			num_chunks += 1
			print(f"Chunk {num_chunks} processed")
		pst.finish_map(barnacle_depth)
		print(f"Preprocessing took {time.time() - start_time} seconds")

		# Psychic Barnacle Streaming
		barn_time = time.time()
		PB = PsychicBarnacle()
		num_chunks = 0
		for i in range(0, len(X), chunk_size):
			chunk = X[i:i+chunk_size]

			# load_df returns OG_X and Y to be used later 
			X_, Y_ = X[i:i+chunk_size], Y[i:i+chunk_size]
			X_processed = pst.use_map(X_)
			# X.to_csv(f"./data/X_cs{chunk_size}_{num_chunks}.csv", index=False)

			# assign_outcome returns outcomes and return_dtype 
			outcomes, return_dtype = assign_outcome(
				X_processed, 
				Y_,
				use_ML=False
			)

			"""Use processed in psychic barnacle, as chunks 
			"""
			col_info = {
				"col_graph": col_graph,
				"cols_dropped": c_dropped
			}
			if num_chunks == 0:  # INITIALIZE	
				PB.initialize_inputs(
					first_X_input = X_processed, 
					col_info=col_info,
					first_Y_input = Y_, 
					first_outcomes_input = outcomes, 
					return_dtype = return_dtype,
					target=target
				)
				# PB.initialize_columns(col_graph, c_dropped)
				PB.make_col_combos(X_processed)

			# Stuff that runs each time 
			PB.vals_map_updating(X_processed, outcomes)
			num_chunks += 1

		PB.finish_map()
		print(f"Barnacle Building took {time.time() - barn_time} seconds for chunk size {chunk_size}")

		num_chunks = 0
		use_time = time.time()
		for i in range(0, len(df), chunk_size):
			"""Append psychic barnacle to each other 
			"""
			# load_df returns OG_X and Y to be used later 
			X_, Y_ = X[i:i+chunk_size], Y[i:i+chunk_size]
			X_processed = pst.use_map(X_)
			outcomes, return_dtype = assign_outcome(X_processed, Y_, use_ML=False)
			processed =PB.use_map( 
				X_processed, 
				# pd.DataFrame(X_, columns=col_graph.values()),
				X_,
				Y_, 
				outcomes
			)
			if num_chunks == 0:
				pb_map = processed
			else:
				# Add to processed
				pb_map = pd.concat([pb_map, processed])
			num_chunks += 1
			print(f"Chunk {num_chunks} processed")
		print(f"Barnacle Mapping took {time.time() - use_time} seconds")

		# Save to csv
		pb_map.to_csv(f"./data/pb_map_{chunk_size}.csv", index=False)

		# Convert PB.vals_map to string values 
		vals_map_avg = PB.vals_map_avg
		val_map_int = {k: str(v) for k, v in PB.val_map_int.items()}
		converted = {} 
		for val_tup in vals_map_avg:
			converted[val_map_int[val_tup[0]], val_map_int[val_tup[1]]] = vals_map_avg[val_tup]

		val_maps[chunk_size] = converted
		check[f"{chunk_size}"] = pb_map

	# check = pd.DataFrame(check)[0:10000]
	temp = {} 
	temp2 = {}   # This can hold a dataframe of column values 
	for chunk in check:
		# other_chunks = [check[other_chunk] for other_chunk in check if other_chunk != chunk]
		other_chunks = [c for c in check if c != chunk]
		# perform a regression on each column
		for other_chunk in other_chunks:
			for col in check[chunk].columns:
				if "_barn" in col:
					temp[col + f"_{chunk}-{other_chunk}"] = r2_score(check[chunk][col], check[other_chunk][col])
					if f"_{chunk}" not in temp2:
						temp2[col + f"_{chunk}"] = list(check[chunk][col])
					if f"_{other_chunk}" not in temp2:
						temp2[col + f"_{other_chunk}"] = list(check[other_chunk][col])
	check = pd.DataFrame(temp, index = [0]).T
	check2 = pd.DataFrame(temp2)

	# Check3 looks at which values are not the same in the vals_map for each 
	check3 = {} 
	for chunk in val_maps:
		other_chunks = [c for c in val_maps if c != chunk]
		for other_chunk in other_chunks:
			for val_tup in val_maps[chunk]:
				if val_tup not in val_maps[other_chunk]:
					check3[f"{chunk}_{other_chunk}_{val_tup}"] = {
						0: val_maps[chunk][val_tup],
						1: "not in"
					}
				else:
					# num1 = f"{val_maps[chunk][val_tup][0]:.4g}"
					# num2 = f"{val_maps[other_chunk][val_tup][0]:.4g}"
					num1 = f"{val_maps[chunk][val_tup]:.4g}"
					num2 = f"{val_maps[other_chunk][val_tup]:.4g}"

					# if val_maps[chunk][val_tup][0] != val_maps[other_chunk][val_tup][0]:
					if num1 != num2:
						check3[f"{chunk}_{other_chunk}_{val_tup}"] = {
							# 0: val_maps[chunk][val_tup][0],
							# 1: val_maps[other_chunk][val_tup][0]
							0: num1,
							1: num2
						}
						debug=1
	
	# Save check3
	check3 = pd.DataFrame(check3).T
	check3.to_csv(f"./data/check3_cython.csv")
	


	# check = pd.DataFrame(check)
	check.to_csv(f"./data/check_cython.csv")
	check2.to_csv(f"./data/check2_cython.csv")

	debug=1


def pb_stream_parquet(path, chunk_sizes):

	# path = "./data/TN_Jerry4.parquet"
	# path = "./data/lum_agg_large.parquet"
	dataset = pq.ParquetDataset(path)
	table = dataset.read()
	test = "full"
	if test == "full":
		# chunk_sizes = [
		# 	len(table) // 5,
		# 	len(table) // 4,
		# ]
		chunk_sizes=chunk_sizes
	else:
		chunk_sizes = [
			1000, 
			1500
		]
		# df = df[:3000]  # Testing 
		df = table.to_pandas()
		df = df[:3000]  # Testing
	target = "std" 

	cols_to_drop = [
		"npi",
		"billing_code",
		"npi_zip5",
		"city",
		"npi_city",
		"census_zip",
		"census_city",
		"census_lat",
		"census_lng",
		"census_county_fips",
		"census_county_name",
		"median",
		# "std",
		"count",
		"family",
		"sub_family",
		# "npi_count",
	]

	barnacle_depth = {
		"main": 8,
	}

	# Define X, X_untouched, Y, col_graph 
	


	val_maps = {} 
	check = {} 
	for chunk_size in chunk_sizes:
		if chunk_size > len(table):
			print("Chunk size is larger than dataframe, skipping")
			sys.exit() 

		# Preprocess
		start_time = time.time()
		pst = preprocess_stream(cols_to_drop=cols_to_drop, target=target)
		num_chunks = 0
		# for i in range(0, len(X), chunk_size):
		# 	chunk = X[i:i+chunk_size]
		for start in range(0, len(table), chunk_size):
			end=start+chunk_size
			df = table.slice(start, end-start).to_pandas()
			X_, Y_, col_graph, c_dropped = load_numpy(df, cols_to_drop, target)
			chunk = X_
			pst.preprocess(chunk, col_graph, c_dropped)
			num_chunks += 1
			print(f"Chunk {num_chunks} processed")
		pst.finish_map(barnacle_depth)
		print(f"Preprocessing took {time.time() - start_time} seconds")

		# Psychic Barnacle Streaming
		barn_time = time.time()
		PB = PsychicBarnacle()
		num_chunks = 0
		# for i in range(0, len(X), chunk_size):
		# 	chunk = X[i:i+chunk_size]
		for start in range(0, len(table), chunk_size):
			end=start+chunk_size
			df = table.slice(start, end-start).to_pandas()
			X_, Y_, col_graph, c_dropped = load_numpy(df, cols_to_drop, target)
			# X_, Y_ = X[i:i+chunk_size], Y[i:i+chunk_size]
			X_processed = pst.use_map(X_)
			# X.to_csv(f"./data/X_cs{chunk_size}_{num_chunks}.csv", index=False)

			# assign_outcome returns outcomes and return_dtype 
			outcomes, return_dtype = assign_outcome(
				X_processed, 
				Y_,
				use_ML=False
			)

			"""Use processed in psychic barnacle, as chunks 
			"""
			col_info = {
				"col_graph": col_graph,
				"cols_dropped": c_dropped
			}
			if num_chunks == 0:  # INITIALIZE	
				PB.initialize_inputs(
					first_X_input = X_processed, 
					col_info=col_info,
					first_Y_input = Y_, 
					first_outcomes_input = outcomes, 
					return_dtype = return_dtype,
					target=target
				)
				# PB.initialize_columns(col_graph, c_dropped)
				PB.make_col_combos(X_processed)

			# Stuff that runs each time 
			PB.vals_map_updating(X_processed, outcomes)
			num_chunks += 1

		PB.finish_map()
		print(f"Barnacle Building took {time.time() - barn_time} seconds for chunk size {chunk_size}")

		num_chunks = 0
		use_time = time.time()
		# for i in range(0, len(df), chunk_size):
		for start in range(0, len(table), chunk_size):
			end=start+chunk_size
			df = table.slice(start, end-start).to_pandas()
			X_, Y_, col_graph, c_dropped = load_numpy(df, cols_to_drop, target)
			X_processed = pst.use_map(X_)
			outcomes, return_dtype = assign_outcome(X_processed, Y_, use_ML=False)
			processed =PB.use_map( 
				X_processed, 
				# pd.DataFrame(X_, columns=col_graph.values()),
				X_,
				Y_, 
				outcomes
			)
			if num_chunks == 0:
				pb_map = processed
			else:
				# Add to processed
				pb_map = pd.concat([pb_map, processed])
			num_chunks += 1
			print(f"Chunk {num_chunks} processed")
		print(f"Barnacle Mapping took {time.time() - use_time} seconds")

		# Save to csv
		# pb_map.to_csv(f"./data/pb_map_{chunk_size}.csv", index=False)
		save_name = f"{path.replace('.parquet', '')}_pb_map_{chunk_size}.csv"
		pb_map.to_csv(save_name, index=False)
		# Save a preview
		pb_map[:1000].to_csv(f"{save_name.replace('.csv', '_preview.csv')}", index=False)

		# Convert PB.vals_map to string values 
		vals_map_avg = PB.vals_map_avg
		val_map_int = {k: str(v) for k, v in PB.val_map_int.items()}
		converted = {} 
		for val_tup in vals_map_avg:
			converted[val_map_int[val_tup[0]], val_map_int[val_tup[1]]] = vals_map_avg[val_tup]

		val_maps[chunk_size] = converted
		check[f"{chunk_size}"] = pb_map

	# check = pd.DataFrame(check)[0:10000]
	temp = {} 
	temp2 = {}   # This can hold a dataframe of column values 
	for chunk in check:
		# other_chunks = [check[other_chunk] for other_chunk in check if other_chunk != chunk]
		other_chunks = [c for c in check if c != chunk]
		# perform a regression on each column
		for other_chunk in other_chunks:
			for col in check[chunk].columns:
				if "_barn" in col:
					temp[col + f"_{chunk}-{other_chunk}"] = r2_score(check[chunk][col], check[other_chunk][col])
					if f"_{chunk}" not in temp2:
						temp2[col + f"_{chunk}"] = list(check[chunk][col])
					if f"_{other_chunk}" not in temp2:
						temp2[col + f"_{other_chunk}"] = list(check[other_chunk][col])
	check = pd.DataFrame(temp, index = [0]).T
	check2 = pd.DataFrame(temp2)

	# Check3 looks at which values are not the same in the vals_map for each 
	check3 = {} 
	for chunk in val_maps:
		other_chunks = [c for c in val_maps if c != chunk]
		for other_chunk in other_chunks:
			for val_tup in val_maps[chunk]:
				if val_tup not in val_maps[other_chunk]:
					check3[f"{chunk}_{other_chunk}_{val_tup}"] = {
						0: val_maps[chunk][val_tup],
						1: "not in"
					}
				else:
					# num1 = f"{val_maps[chunk][val_tup][0]:.4g}"
					# num2 = f"{val_maps[other_chunk][val_tup][0]:.4g}"
					num1 = f"{val_maps[chunk][val_tup]:.4g}"
					num2 = f"{val_maps[other_chunk][val_tup]:.4g}"

					# if val_maps[chunk][val_tup][0] != val_maps[other_chunk][val_tup][0]:
					if num1 != num2:
						check3[f"{chunk}_{other_chunk}_{val_tup}"] = {
							# 0: val_maps[chunk][val_tup][0],
							# 1: val_maps[other_chunk][val_tup][0]
							0: num1,
							1: num2
						}
						debug=1
	
	# Save check3
	check3 = pd.DataFrame(check3).T
	check3.to_csv(f"./data/check3_cython.csv")
	


	# check = pd.DataFrame(check)
	check.to_csv(f"./data/check_cython.csv")
	check2.to_csv(f"./data/check2_cython.csv")

	debug=1


def return_barn(table, target, cols_to_drop, barnacle_depth):
	"""
	"""

	chunk_size = 50000
	if len(table) // 5 < chunk_size:
		chunk_size = len(table) // 5

	# Preprocess
	start_time = time.time()
	pst = preprocess_stream(cols_to_drop=cols_to_drop, target=target)
	num_chunks = 0
	for start in range(0, len(table), chunk_size):
		end=start+chunk_size
		df = table.slice(start, end-start).to_pandas()
		X_, Y_, col_graph, c_dropped = load_numpy(df, cols_to_drop, target)
		chunk = X_
		pst.preprocess(chunk, col_graph, c_dropped)
		num_chunks += 1
		print(f"Chunk {num_chunks} processed")
	pst.finish_map(barnacle_depth)

	# Psychic Barnacle Streaming
	barn_time = time.time()
	PB = PsychicBarnacle()
	num_chunks = 0
	for start in range(0, len(table), chunk_size):
		end=start+chunk_size
		df = table.slice(start, end-start).to_pandas()
		X_, Y_, col_graph, c_dropped = load_numpy(df, cols_to_drop, target)
		X_processed = pst.use_map(X_)

		outcomes, return_dtype = assign_outcome(
			X_processed, 
			Y_,
			use_ML=False
		)

		"""Use processed in psychic barnacle, as chunks 
		"""
		col_info = {
			"col_graph": col_graph,
			"cols_dropped": c_dropped
		}
		if num_chunks == 0:  # INITIALIZE	
			PB.initialize_inputs(
				first_X_input = X_processed, 
				col_info=col_info,
				first_Y_input = Y_, 
				first_outcomes_input = outcomes, 
				return_dtype = return_dtype,
				target=target
			)
			PB.make_col_combos(X_processed)

		# Stuff that runs each time 
		PB.vals_map_updating(X_processed, outcomes)
		num_chunks += 1
	PB.finish_map()

	num_chunks = 0
	use_time = time.time()
	for start in range(0, len(table), chunk_size):
		end=start+chunk_size
		df = table.slice(start, end-start).to_pandas()
		X_, Y_, col_graph, c_dropped = load_numpy(df, cols_to_drop, target)
		X_processed = pst.use_map(X_)
		outcomes, return_dtype = assign_outcome(X_processed, Y_, use_ML=False)
		processed =PB.use_map( 
			X_processed, 
			X_,
			Y_, 
			outcomes
		)
		if num_chunks == 0:
			pb_map = processed
		else:
			# Add to processed
			pb_map = pd.concat([pb_map, processed])
		num_chunks += 1
		print(f"Chunk {num_chunks} processed")

	return pb_map


if __name__ == "__main__":

	# test_preprocess()

	# pb_stream_algorithm() 

	# path = "./data/lum_agg_large.parquet"
	path = "./data/train_engineered_skincancer.parquet"
	# pb_stream_parquet(path, chunk_sizes = [50000, 80000])

	# path = "./data/lum_agg_large.parquet"
	dataset = pq.ParquetDataset(path)
	table = dataset.read()
	# target = "std"
	target="target"
	cols_to_drop = [
		"sex",
		"anatom_site_general",
		"tbp_tile_type",
		"tbp_lv_location",
		"tbp_lv_location_simple",
		"attribution"
		# "npi",
		# "billing_code",
		# "npi_zip5",
		# "city",
		# "npi_city",
		# "census_zip",
		# "census_city",
		# "census_lat",
		# "census_lng",
		# "census_county_fips",
		# "census_county_name",
		# "median",
		# # "std",
		# "count",
		# "family",
		# "sub_family",
		# "npi_count",
	]
	barnacle_depth = {
		"main": 8,
	}
	pb_map = return_barn(table, target, cols_to_drop, barnacle_depth)

	debug=1
