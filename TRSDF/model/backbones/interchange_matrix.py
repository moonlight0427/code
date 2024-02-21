import numpy as np

class interchange_matrix():
    def __init__(self, non_cls):  # 要传去掉class_token的patch进来获取
        B, N, C = non_cls.size()
        W = int((N / 2.0) ** 0.5)  # W：11
        H = int(N / W)  # H：22






    def generate_interchange_matrix(x):
        # matrix = np.identity(2)
        B, N, C = x.shape
        num_patches = N - 1
        matrix = np.diag([1] * 242)
        return matrix



if __name__ == '__main__':
    A = interchange_matrix.generate_interchange_matrix()
    print(A)