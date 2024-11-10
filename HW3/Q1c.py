import numpy as np
import math

def euclidean_dist(pt, mu, i):
    dist = math.sqrt((pt[0]-mu[0])**2 + (pt[1]-mu[1])**2)
    print(f"d({pt}, $\\mu_{i}$) = $\sqrt" + "{" + f"{pt[0]}-{mu[0]})^2 + ({pt[1]}-{mu[1]})^2)" + "}" + f"={dist}" + "$ \\newline")

def inf_norm(pt, mu, i):
    dist = max(abs(pt[0]-mu[0]), abs(pt[1]-mu[1]))
    print(f"d({pt}, $\\mu_{i}$) = max(|{pt[0]}-{mu[0]}|, |{pt[1]}-{mu[1]}|)={dist} \\newline")

def main():
    mu = np.array([[(110/3), (110/3)], [(48/4), (-40/4)], [(29/3), (-13/3)]])

    pts = np.array([[0, 0], [10, 10], [100, 100], [50, -50], [30, -4], [1, 1], [0, 5], [-3, 4], [-1, 1], [0, 10]])

    for pt in pts:
        print(f"For {pt}: \\newline")
        for i in range(3):
            inf_norm(pt, mu[i], i)

if __name__=="__main__":
    main()