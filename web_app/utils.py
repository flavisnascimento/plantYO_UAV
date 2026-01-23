import math
# from python_tsp.exact import solve_tsp_dynamic_programming
import numpy as np

METERS_IN_DEGREE = 111319.5  # Approximate meters per degree latitude

def gps_to_local_xy(lat, lon, origin_lat, origin_lon):
   
    meters_in_long_degree = math.cos(math.radians(origin_lat)) * METERS_IN_DEGREE

    x = (lon - origin_lon) * meters_in_long_degree
    y = (lat - origin_lat) * METERS_IN_DEGREE

    return float(x), float(y)


def distance(p1, p2):
    dx = p1['x'] - p2['x']
    dy = p1['y'] - p2['y']
    return math.hypot(dx, dy)


# def tsp_solver(points, start_lat, start_lon, origin_lat, origin_lon):
#     # Convert drone start position
#     start_x, start_y = gps_to_local_xy(start_lat, start_lon, origin_lat, origin_lon)
#     start_point = {'name': 'START', 'x': start_x, 'y': start_y}

#     # Insert start point at beginning of list
#     full_points = [start_point] + points

#     # Build distance matrix
#     n = len(full_points)
#     dist_matrix = np.zeros((n, n))
#     for i in range(n):
#         for j in range(n):
#             dist_matrix[i][j] = distance(full_points[i], full_points[j])

#     # Solve TSP including the start
#     perm, total_dist = solve_tsp_dynamic_programming(dist_matrix)

#     # Remove the dummy START from result
#     perm = [i for i in perm if i != 0]  # remove index 0 (the start)
#     ordered_points = [points[i - 1] for i in perm]  # adjust index to match original points

#     return ordered_points
