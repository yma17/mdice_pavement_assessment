"""
Define polygon for Detroit city limits.
References:
https://detroitmi.gov/webapp/interactive-district-map
https://epsg.io/map

Borders are defined fairly conservatively w.r.t. actual city limits.
"""

from shapely.geometry.polygon import Polygon

import pickle


def main():
	outer_border = get_outer_border()
	inner_border = get_inner_border()
	border_info = {"outer": outer_border, "inner": inner_border}
	
	# Write borders to pickle file.
	with open('../../data/derived/detroit_borders.pickle', 'wb') as f:
		pickle.dump(border_info, f)


def get_outer_border():
	"""Retrieve a polygon representing the 'outer' border of Detroit."""

	outer_border_coords = []  # stores (long, lat pairs) - e.g. (-83, 42)

	# Append vertices.
	outer_border_coords.append((-83.098183, 42.286897))
	outer_border_coords.append((-83.118074, 42.289572))
	outer_border_coords.append((-83.119683, 42.287215))
	outer_border_coords.append((-83.117280, 42.279023))
	outer_border_coords.append((-83.129253, 42.280262))
	outer_border_coords.append((-83.137515, 42.282786))
	outer_border_coords.append((-83.161139, 42.254697))
	outer_border_coords.append((-83.163049, 42.256904))
	outer_border_coords.append((-83.164101, 42.257682))
	outer_border_coords.append((-83.166997, 42.259525))
	outer_border_coords.append((-83.167341, 42.261875))
	outer_border_coords.append((-83.168414, 42.263971))
	outer_border_coords.append((-83.173349, 42.265051))
	outer_border_coords.append((-83.167641, 42.267862))
	outer_border_coords.append((-83.158425, 42.278682))
	outer_border_coords.append((-83.162041, 42.281945))
	outer_border_coords.append((-83.164465, 42.286580))
	outer_border_coords.append((-83.167255, 42.288913))
	outer_border_coords.append((-83.167856, 42.290596))
	outer_border_coords.append((-83.165474, 42.290548))
	outer_border_coords.append((-83.158865, 42.292247))
	outer_border_coords.append((-83.157320, 42.293739))
	outer_border_coords.append((-83.156569, 42.295580))
	outer_border_coords.append((-83.151569, 42.296564))
	outer_border_coords.append((-83.143823, 42.293390))
	outer_border_coords.append((-83.143866, 42.294469))
	outer_border_coords.append((-83.142707, 42.294469))
	outer_border_coords.append((-83.141613, 42.295167))
	outer_border_coords.append((-83.141055, 42.296008))
	outer_border_coords.append((-83.140604, 42.296881))
	outer_border_coords.append((-83.140283, 42.298199))
	outer_border_coords.append((-83.140154, 42.299072))
	outer_border_coords.append((-83.140304, 42.299818))
	outer_border_coords.append((-83.141313, 42.302055))
	outer_border_coords.append((-83.141656, 42.303833))
	outer_border_coords.append((-83.141913, 42.304928))
	outer_border_coords.append((-83.142707, 42.305801))
	outer_border_coords.append((-83.140583, 42.306880))
	outer_border_coords.append((-83.140841, 42.307768))
	outer_border_coords.append((-83.139617, 42.308768))
	outer_border_coords.append((-83.140433, 42.310529))
	outer_border_coords.append((-83.153651, 42.327728))
	outer_border_coords.append((-83.156826, 42.326824))
	outer_border_coords.append((-83.157256, 42.330139))
	outer_border_coords.append((-83.157620, 42.337262))
	outer_border_coords.append((-83.153372, 42.337833))
	outer_border_coords.append((-83.151119, 42.339117))
	outer_border_coords.append((-83.150175, 42.340029))
	outer_border_coords.append((-83.149488, 42.341100))
	outer_border_coords.append((-83.147857, 42.349624))
	outer_border_coords.append((-83.148029, 42.351297))
	outer_border_coords.append((-83.195429, 42.349664))
	outer_border_coords.append((-83.194828, 42.335882))
	outer_border_coords.append((-83.211930, 42.335691))
	outer_border_coords.append((-83.213561, 42.335025))
	outer_border_coords.append((-83.214977, 42.335580))
	outer_border_coords.append((-83.213239, 42.327427))
	outer_border_coords.append((-83.225706, 42.328331))
	outer_border_coords.append((-83.227744, 42.331519))
	outer_border_coords.append((-83.235576, 42.328664))
	outer_border_coords.append((-83.236392, 42.335104))
	outer_border_coords.append((-83.238065, 42.335200))
	outer_border_coords.append((-83.238602, 42.342496))
	outer_border_coords.append((-83.242314, 42.342511))
	outer_border_coords.append((-83.253644, 42.341163))
	outer_border_coords.append((-83.264716, 42.340925))
	outer_border_coords.append((-83.267591, 42.357053))
	outer_border_coords.append((-83.268256, 42.378329))
	outer_border_coords.append((-83.276324, 42.378012))
	outer_border_coords.append((-83.279500, 42.405999))
	outer_border_coords.append((-83.288426, 42.405967))
	outer_border_coords.append((-83.289735, 42.443538))
	outer_border_coords.append((-83.259287, 42.446071))
	outer_border_coords.append((-83.219891, 42.447528))
	outer_border_coords.append((-83.165860, 42.447718))
	outer_border_coords.append((-83.126335, 42.448478))
	outer_border_coords.append((-83.095179, 42.449903))
	outer_border_coords.append((-83.044667, 42.450853))
	outer_border_coords.append((-83.000293, 42.452151))
	outer_border_coords.append((-82.966304, 42.452215))
	outer_border_coords.append((-82.936392, 42.452563))
	outer_border_coords.append((-82.948623, 42.436602))
	outer_border_coords.append((-82.926435, 42.427606))
	outer_border_coords.append((-82.908454, 42.420700))
	outer_border_coords.append((-82.908926, 42.415283))
	outer_border_coords.append((-82.912445, 42.407298))
	outer_border_coords.append((-82.916822, 42.398678))
	outer_border_coords.append((-82.921329, 42.393354))
	outer_border_coords.append((-82.934246, 42.388917))
	outer_border_coords.append((-82.942615, 42.385684))
	outer_border_coords.append((-82.923775, 42.357656))
	outer_border_coords.append((-82.947979, 42.344970))
	outer_border_coords.append((-82.957850, 42.336786))
	outer_border_coords.append((-82.986689, 42.331012))
	outer_border_coords.append((-83.017588, 42.329552))
	outer_border_coords.append((-83.063164, 42.317939))
	outer_border_coords.append((-83.078699, 42.308482))
	outer_border_coords.append((-83.096638, 42.289628))

	outer_border = Polygon(outer_border_coords)
	return outer_border


def get_inner_border():
	"""
	Retrieve a polygon representing the 'inner border of Detroit.
	(aka, the borders of the enclaves of Highland Park and Hamtramck
	with Detroit)
	"""
	inner_border_coords = []

	# Append vertices.
	inner_border_coords.append((-83.118804, 42.415885))
	inner_border_coords.append((-83.100779, 42.391230))
	inner_border_coords.append((-83.072391, 42.400184))
	inner_border_coords.append((-83.065588, 42.391880))
	inner_border_coords.append((-83.061211, 42.393465))
	inner_border_coords.append((-83.053379, 42.383401))
	inner_border_coords.append((-83.044281, 42.386222))
	inner_border_coords.append((-83.046727, 42.390977))
	inner_border_coords.append((-83.044152, 42.392150))
	inner_border_coords.append((-83.043809, 42.403274))
	inner_border_coords.append((-83.054967, 42.402798))
	inner_border_coords.append((-83.058014, 42.407013))
	inner_border_coords.append((-83.077970, 42.400612))
	inner_border_coords.append((-83.091574, 42.416170))

	inner_border = Polygon(inner_border_coords)
	return inner_border


if __name__ == '__main__':
	main()