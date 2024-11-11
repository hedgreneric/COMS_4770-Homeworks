import numpy as np

class Linear_System_Solver():
    def LUdcmp(self, A):
        n = len(A)
        L = np.zeros((n, n))
        U = np.zeros((n, n))
        P = np.eye(n)

        # Pivoting
        for i in range(n):
            max_row = np.argmax(np.abs(A[i:n, i])) + i
            if i != max_row:
                A[[i, max_row]] = A[[max_row, i]]
                P[[i, max_row]] = P[[max_row, i]]
        
        # Crout's Algorithm with Pivoting
        for i in range(n):
            for j in range(i, n):
                L[j][i] = A[j][i] - sum(L[j][k] * U[k][i] for k in range(i))
            for j in range(i + 1, n):
                U[i][j] = (A[i][j] - sum(L[i][k] * U[k][j] for k in range(i))) / L[i][i]
            U[i][i] = 1

        return L, U, P

    def LUbksub(self, L, U, P, b):
        n = len(L)
        
        # Forward substitution
        b = np.dot(P, b)
        y = np.zeros_like(b)
        for i in range(n):
            y[i] = b[i] - np.dot(L[i, :i], y[:i])
        
        # Backward substitution
        x = np.zeros_like(b)
        for i in range(n-1, -1, -1):
            x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
        
        return x

    def solve_system(self, A, b):
        L, U, P = self.LUdcmp(np.array(A, dtype=float))
        x = self.LUbksub(L, U, P, np.array(b, dtype=float))
        return x

