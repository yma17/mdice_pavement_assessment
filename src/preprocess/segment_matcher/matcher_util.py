"""
File containing utility functions for segment matcher.
"""

import math


def line_intersection(line1, line2):
    """Source: https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines"""
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def great_circle_distance(pt1_long, pt1_lat, pt2_long, pt2_lat, radius):
    """
    Uses Haversine formula to compute great circle distance.
    Source: https://en.wikipedia.org/wiki/Great-circle_distance
    """
    phi_1 = math.radians(pt1_lat)
    phi_2 = math.radians(pt2_lat)
    delta_phi = abs(phi_1 - phi_2)
    delta_lambda = abs(math.radians(pt1_long) - math.radians(pt2_long))

    sqrt_term1 = math.sin(delta_phi / 2) ** 2
    sqrt_term2 = math.cos(phi_1) * math.cos(phi_2)
    sqrt_term3 = math.sin(delta_lambda / 2) ** 2
    sqrt_term = math.sqrt(sqrt_term1 + sqrt_term2 * sqrt_term3)

    return 2 * radius * math.asin(sqrt_term)


def intersect(nums1, nums2):
    """
    Source: https://stackoverflow.com/questions/37645053/intersection-of-two-lists-including-duplicates
    """
    match = {}
    for x in nums1:
        if x in match:
            match[x] += 1
        else:
            match[x] = 1

    c = []
    for x in nums2:
        if x in match:
            c.append(x)
            match[x] -= 1
            if match[x] == 0:
                del match[x]

    return c


def get_overlap(hash1, hash2):
    """Compute overlap between two hashes."""
    return len(hash1.intersection(hash2))