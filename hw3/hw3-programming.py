import numpy as np
import time

start_time = time.time_ns()

l_input = input("Enter the elements of vector l, separated by spaces: ")
l = np.array([int(x) for x in l_input.split()])

v_input = input("Enter the elements of vector v, separated by spaces: ")
v = np.array([int(x) for x in v_input.split()])

vertices = np.array([
    [0, 0, 0, 1],  # Vertex 1
    [1, 0, 0, 1],  # Vertex 2
    [0, 1, 0, 1],  # Vertex 3
    [1, 1, 1, 1]   # Vertex 4
])

projection_matrix = np.outer(v, l) - np.dot(l, v)*np.identity(4)

def project_points(points, matrix):
    projected_points = []
    for point in points:
        projected_point = matrix @ point  # Matrix multiplication
        projected_points.append(projected_point)
    return np.array(projected_points)


# Project the tetrahedron vertices
projected_vertices = project_points(vertices, projection_matrix)

end_time = time.time_ns()

runtime = end_time - start_time

print("\nRuntime: " + str(runtime) + " ns")

# Output the projection matrix and projected points
print("\nProjection Matrix:")
print(projection_matrix)

print("\nProjected Vertices:")
print(projected_vertices)
