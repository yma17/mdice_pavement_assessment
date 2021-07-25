"""
File containing high-level functions for MRB decision process.
"""

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import random


class MRB():
	def __init__(self, config):
		"""Initialize parameters for model/algorithm."""
		self.c_mrb = config['c_mrb']  # road length constraint

		w_benefit, w_quality = config['benefit_pref'], config['quality_pref']
		# (normalized) benefit dimension weight:
		self.w_benefit = w_benefit / (w_benefit + w_quality)
		# (normalized) quality dimension weight:
		self.w_quality = w_quality / (w_benefit + w_quality)

		w_volume, w_bus = config['volume_pref'], config['bus_pref']
		# (normalized) traffic volume weight:
		self.w_volume = w_volume / (w_volume + w_bus)
		# (normalized) bus volume weight:
		self.w_bus = w_bus / (w_volume + w_bus)

		self.alpha_q = config['a_q']  # alpha for road quality value sigmoid curve
		self.alpha_p = config['a_p']  # alpha for ellipse boundary sigmoid curve

		# PASER rating bounds
		self.min_q = 1
		self.max_q = config['max_q']  # don't repair better quality roads than this

		# Vicinity selection criteria
		self.k = config['k']


	def calc_boundary_val(self, l2_norm, curr_val):
		return 1.0 / (1 + np.exp(-1 * self.alpha_p * (l2_norm - curr_val)))


	def calc_quality_val(self, quality):
		return 1.0 / (1 + np.exp(self.alpha_q * (quality - self.q_mean)))


	def calc_benefit_val(self, aadt, bus_ridership):
		return self.w_volume * aadt + self.w_bus * bus_ridership


	def calc_weighted_l2norm(self, b_val, qv_val):
		return (self.w_benefit * b_val**2 + self.w_quality * qv_val**2) ** 0.5


	def impl(self):
		"""Implementation of algorithm."""

		## Load csv containing road quality, traffic volume, bus ridership
		##	for all road segments to consider.
		df = pd.read_csv("../data/derived/mrb_data_paser.csv")
		df_orig = df.copy()

		## Filter segments according to PASER rating
		df = df[df["paser_rating"] >= self.min_q]
		df = df[df["paser_rating"] <= self.max_q]

		## For use in calc_quality_val
		self.q_mean = (self.min_q + self.max_q) / 2.0

		## Normalize aadt, bus_ridership columns.
		df['aadt'] -= df['aadt'].min()
		df['aadt'] /= df['aadt'].max()
		df['adr'] -= df['adr'].min()
		df['adr'] /= df['adr'].max()

		## Compute quality values for each road segment.
		## These remain unchanged throughout iterations of algorithm.
		quality_val_list, l2_norm_list = [], []
		for _, row in df.iterrows():
			quality_val = self.calc_quality_val(row["paser_rating"])
			quality_val_list.append(quality_val)
		df['quality_val'] = quality_val_list

		#####
		## Run iterations of greedy knapsack until complete.
		#####
		length_repaired = 0
		selected, selected_scores = [], []
		#visualized = False
		while len(df) > 0 and length_repaired <= self.c_mrb:
			## Compute benefit values for each road segment.
			## Due to overlap, these will change throughout iterations.
			benefit_val_list = []
			for _, row in df.iterrows():
				benefit_val = self.calc_benefit_val(row["aadt"], row["adr"])
				benefit_val_list.append(benefit_val)
			df['benefit_val'] = benefit_val_list
			df['benefit_val'] -= df['benefit_val'].min()  # normalize
			df['benefit_val'] /= df['benefit_val'].max()  # normalize

			## Compute weighted L2 norms of each road segment.
			l2_norm_list = []
			for _, row in df.iterrows():
				l2_norm = self.calc_weighted_l2norm(row["benefit_val"],
													row["quality_val"])
				l2_norm_list.append(l2_norm)
			df['weighted_l2norm'] = l2_norm_list
			max_l2norm = max(df['weighted_l2norm'])

			# ## Visualize the points of the first iteration
			# if not visualized:
			# 	fig = plt.figure()
			# 	ax = fig.add_subplot(111, projection='3d')
			# 	ax.scatter(df['benefit_val'], df['quality_val'], df['weighted_l2norm'])
			# 	ax.set_xlabel("Benefit")
			# 	ax.set_ylabel("Quality")
			# 	ax.set_zlabel("Weighted L2 norm")
			# 	#plt.show()
			# 	visualized = True

			## Compute probabilities for selecting each road segment.
			prob_list, prob_sum = [], 0.0
			for _, row in df.iterrows():
				prob = self.calc_boundary_val(row['weighted_l2norm'], max_l2norm)
				prob_sum += prob
				prob_list.append(prob)
			df['prob'] = prob_list
			df['prob'] /= prob_sum  # normalize
			#print(df[['weighted_l2norm', 'prob']])
			#print()

			## Sort in descending order according to l2 norm.
			df = df.sort_values(by=['weighted_l2norm'], ascending=False)

			## Probabilistically select a segment
			## Then, select k segments that are geographically closest.
			## Update df + length_repaired on each step.
			prob_val = random.random()
			prob_cumul, sel_i, sel_row = 0, None, df.iloc[0]
			for i, row in df.iterrows():
				prob_cumul += row["prob"]
				if prob_val <= prob_cumul:
					sel_row = row
					sel_i = i
					break
			#print(df.loc[sel_i][['weighted_l2norm', 'prob']])

			selected.append(sel_row["lrs_link"])
			selected_scores.append(sel_row["weighted_l2norm"])
			highest_pm_dist = sel_row["pm_dist_km"]
			highest_eq_dist = sel_row["eq_dist_km"]
			length_repaired += sel_row["length"]

			try:
				df = df.drop([sel_i])
			except KeyError:  # possibly due to floating point issues
				df = df.iloc[1:]
			
			df["dist_from_sel"] = (highest_pm_dist - df["pm_dist_km"]) ** 2 + \
								  (highest_eq_dist - df["eq_dist_km"]) ** 2
			df = df.sort_values(by=['dist_from_sel'], ascending=True)
			i = 0
			while i < min(self.k, len(df)) and length_repaired <= self.c_mrb:
				selected.append(df.iloc[i]["lrs_link"])
				selected_scores.append(df.iloc[i]["weighted_l2norm"])
				length_repaired += df.iloc[i]["length"]
				i += 1
			try:
				df = df.iloc[self.k:]
			except IndexError:  # if run out of dataframe rows
				df = df.iloc[0:0]  # then drop all rows

		## Save info of segments selected
		df_selected = pd.DataFrame(dict.fromkeys(list(df_orig.columns), []))
		for segment in selected:
			df_seg = df_orig[df_orig["lrs_link"].isin([segment])]
			df_selected = df_selected.append(df_seg, ignore_index=True)
		df_selected = df_selected.drop(['pm_dist_km', 'eq_dist_km'], axis=1)
		df_selected["decision_score"] = selected_scores
		df_selected.set_index('lrs_link', inplace=True)
		df_selected.to_csv("../output/mrb_results.csv", index=True)


def mrb_proc(config):
	print("--- RUNNING MAJOR ROAD DECISION PROCESS (step 6) ---\n")

	mrb = MRB(config)
	mrb.impl()