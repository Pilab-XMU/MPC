import numpy as np
class MPVC(object):
    def __init__(self, n_cluster = 2, epoch = 100, error=1e-6, sigma = 1, m=2):
        self.K = n_cluster #聚类数
        self.epoch = epoch # 迭代轮数
        self.error = error 
        self.sigma = sigma # 超参数，控制距离的归一化
        self.m = m # 模糊参数
    def fit(self, cond, ref):
        self.cond = cond
        self.ref = ref
        self.N = cond.shape[1] # 轨迹长度
        self.M = cond.shape[0] # 样本数
        self.trans() # 特征变幻
        
        centers = []
        u = np.random.dirichlet(np.ones(self.M), size=self.K)
        e = 0
        while e < self.epoch:
            u2 = u.copy()
            centers = self.next_centers(self.X, u)
            f = self._covariance(self.X, centers, u)
            dist = self._distance(self.X, centers, f)
            u = self.next_u(dist)
            e += 1

            # Stopping rule
            if np.linalg.norm(u - u2) < self.error:
                break
        self.f = f
        self.u = u
        self.centers = centers
        self.labels_ = self.predict(self.X)
        
    def trans(self):
        self.Y = self.cond - self.ref
        self.X_delta = np.linalg.norm(self.Y, axis=1) / np.sqrt(self.N) # 特征1 欧式距离
        self.cos = self.cosine() # 特征2 余弦角
        self.h = self.ham() # 特征3 汉明距离
        self.X = np.column_stack([self.X_delta, self.cos, self.h])
    
    def cosine(self):
        dot_p = np.dot(self.Y, self.ref)
        Y_norm = np.linalg.norm(self.Y, axis=1)
        ref_norm = np.linalg.norm(self.ref)
        sim = -dot_p / (Y_norm * ref_norm)
        return sim
    def ham(self):
        Yr = np.sign(self.Y)
        r = np.sign(self.ref)
        d = np.sum(Yr != r, axis=1)
        return d
    def next_centers(self, X, u):
        um = u ** self.m
        return ((um @ X).T / um.sum(axis=1)).T
    
    def _covariance(self, z, v, u):
        um = u ** self.m

        denominator = um.sum(axis=1).reshape(-1, 1, 1)
        temp = np.expand_dims(z.reshape(z.shape[0], 1, -1) - v.reshape(1, v.shape[0], -1), axis=3)
        temp = np.matmul(temp, temp.transpose((0, 1, 3, 2)))
        numerator = um.transpose().reshape(um.shape[1], um.shape[0], 1, 1) * temp
        numerator = numerator.sum(0)
        return numerator / denominator
    def _distance(self, z, v, f):
        dif = np.expand_dims(z.reshape(z.shape[0], 1, -1) - v.reshape(1, v.shape[0], -1), axis=3)
        determ = np.power(np.linalg.det(f), 1 / self.m)
        det_time_inv = determ.reshape(-1, 1, 1) * np.linalg.pinv(f)
        temp = np.matmul(dif.transpose((0, 1, 3, 2)), det_time_inv)
        output = np.matmul(temp, dif).squeeze().T
        return np.fmax(output, 1e-8)
    
    def next_u(self, d):
        power = float(1 / (self.m - 1))
        d = d.transpose()
        denominator_ = d.reshape((d.shape[0], 1, -1)).repeat(d.shape[-1], axis=1)
        denominator_ = np.power(d[:, None, :] / denominator_.transpose((0, 2, 1)), power)
        denominator_ = 1 / denominator_.sum(1)
        denominator_ = denominator_.transpose()

        return denominator_
    def predict(self, z):
        if len(z.shape) == 1:
            z = np.expand_dims(z, axis=0)

        dist = self._distance(z, self.centers, self.f)
        if len(dist.shape) == 1:
            dist = np.expand_dims(dist, axis=0)

        u = self.next_u(dist)
        return np.argmax(u, axis=0)