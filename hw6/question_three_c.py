from linear_system_solver import Linear_System_Solver

A = [
    [11, 2, -5, 6, 48],
    [1, 0, 17, 29, -21],
    [-3, 4, 55, -61, 0],
    [41, 97, -32, 47, 23],
    [-6, 9, -4, -8, 50]
]

b1 = [4, 0, -7, -2, -11]

b2 = [2, 77, -1003, -7, 10]

lss = Linear_System_Solver()

x1 = lss.solve_system(A, b1)
x2 = lss.solve_system(A, b2)

print(f"Solution for b1: {x1}")
print(f"Solution for b2: {x2}")