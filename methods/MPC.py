import numpy as np
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import pairwise_distances


def get_IOR(x, up_lim, low_lim):
    up_idx = np.where(x > up_lim)[0]
    if len(up_idx) > 0:
        start = up_idx[-1]
    else:
        start = 0

    low_idx = np.where(x < low_lim)[0]
    if len(low_idx) > 0:
        end = low_idx[0]
    else:
        end = len(x)-1
    x_copy = x.copy()[start:end+1]
    x_copy[0] = up_lim
    x_copy[-1] = low_lim
    return x_copy

def normTS(X):
    X_new = []
    for x in X:
        X_new.append((x-x.min()) / (x.max() - x.min()))
    return np.array(X_new)


def distance_2(a, b):
    return np.sum((a-b)**2)

def distance_e(a, b):
    return np.linalg.norm(a-b)


def beta(x, win, comp, dist_fun=distance_2):
    a = dist_fun(comp, x) - dist_fun(win, x)
    b = dist_fun(win, comp)
    return np.exp((-a)/b)

def d_dtw(x, y):
    return fastdtw(x, y)[0]

class MPC(object):
    def __init__(self, K0=2, epoch=200, alpha=0.005, gamma=None, xi=0.001, percent=2, theta=10, K=2):
        self.K0 = K0
        self.epoch = epoch
        self.alpha = alpha 
        self.gamma = gamma 
        self.xi = xi 
        self.percent = percent 
        self.theta = theta 
        self.K = K
    
    def fit(self, X, cond_range=(-5. -0.3),dist=None):
        self.data = X 
        if dist is not None:
            self.dist = dist
        else:
            self.dist = pairwise_distances(X)
        self.local_density = self.getDensityList()
        np.random.seed(1024)
        self.rand_idx = np.arange(len(X))
        np.random.shuffle(self.rand_idx)
        self.diffs = []
        self.split()
        self.merge_v2(K = self.K)

    def cutDistance(self):
        n = len(self.dist)
        dc_list = np.zeros(n)
        cut_num = round(self.percent / 100 * n)
        for i in range(n):
            temp = sorted(self.dist[i])
            dc_list[i] = np.mean(temp[:cut_num])
        return np.mean(dc_list)
    
    def getDensityList(self):
        dc = self.cutDistance()
        neighbor_graph = np.where(self.dist < dc, 1, 0)
        return np.sum(neighbor_graph, axis=0)
    
    def split(self):
        N = len(self.data) 
        gamma = 1 / N if self.gamma is None else self.gamma
        K = self.K0 
        init_idx = self.rand_idx[:K]
        centers = self.data[init_idx]
        centers_old = centers.copy()
        for e in range(self.epoch):
            n_win = np.zeros(K) 
            count = 0
            for i in self.rand_idx:
                x_i = self.data[i]
                dist_centers = np.sum((centers - x_i)**2, axis=1)
                assert(len(dist_centers) == K)
                win_idx = np.argmin(dist_centers)
                n_win[win_idx] += 1
 
                bt_list = []
                for c in range(K):
                    if c != win_idx:
                        bt = beta(x_i, centers[win_idx], centers[c], distance_2)
                        bt_list.append(bt)
                        # centers[c] = centers[c] - K * self.alpha * self.gamma * bt * (x_i - centers[c])
                        centers[c] = centers[c] -  K*self.alpha * gamma * bt * (x_i - centers[c])
                centers[win_idx] = centers[win_idx] +  K*self.alpha * (x_i - centers[win_idx])
                count += 1
            diff = np.sum((centers[n_win> 0] - centers_old[n_win > 0])**2, axis=1)
            max_diff = diff.max()
            self.diffs.append(max_diff)
            print(f'eoch={e}, diff={max_diff}, id={np.argmax(diff)}')
            if (max_diff < self.xi) or ((len(self.diffs) > 10) and (len(np.unique(self.diffs[-10:]))==1)) or (e == self.epoch-1): # 收敛
                d = pairwise_distances(centers, self.data) 
                c_pred = np.argmin(d, axis=0) 
                assert(len(c_pred) == N)
                count = np.array([np.sum(c_pred == k) for k in range(K)])
                die_idx = np.where(count <= self.theta)[0]
                if len(die_idx) > 0: 
                    break
                np.random.shuffle(self.rand_idx) 
                max_gap_idx = self._density_gap(c_pred, K)
                #print(f"分裂{max_gap_idx}")
                center_copy = centers[max_gap_idx].copy() + np.ones(centers.shape[1]) * 1e-4
                center_copy = center_copy.reshape(1, -1)
                K += 1
                centers = np.append(centers, center_copy, axis=0)
            centers_old = centers.copy()
        centers_end = []
        for i in range(len(centers)):
            if i not in die_idx.tolist():
                centers_end.append(centers[i])
        self.centers= np.array(centers_end)
    def merge(self, _bins=100, _range=(0, 1), K=2):
        labels_sb = self._predict()
        self.labels_sb = labels_sb
        dtw_pd = pdist(self.centers, d_dtw)
        dtw_pd = dtw_pd / np.max(dtw_pd) 
        hists = []
        for i in range(len(self.centers)):
            cond = self.data[labels_sb == i]
            hists.append(np.histogram(np.concatenate(cond), bins=_bins, range= _range, density=True)[0])
        hist_pd = pdist(hists)
        hist_pd = hist_pd / np.max(hist_pd)
        d_pd = dtw_pd + hist_pd
        linked = linkage(d_pd, method='average')
        self._linked = linked
        centers_label = fcluster(linked, t = K, criterion='maxclust')-1
        self.labels = np.zeros(len(labels_sb))
        for i, label in enumerate(labels_sb):
            self.labels[i] = centers_label[label]
        self.merge_cost = linked[:, 2][::-1]
        print("merge  end.")
    def merge_v2(self, _bins=100, _range=(0.1, 1), K=2):
        labels_sb = self._predict()
        self.labels_sb = labels_sb
        e_pd = pdist(self.centers)
        e_pd = e_pd / np.max(e_pd) 
        hists = []
        for i in range(len(self.centers)):
            cond = self.data[labels_sb == i]
            hists.append(np.histogram(np.concatenate(cond), bins=_bins, range= _range)[0] / len(self.centers))
        hist_pd = pdist(hists)
        hist_pd = hist_pd / np.max(hist_pd)
        d_pd = e_pd + hist_pd
        linked = linkage(d_pd, method='average')
        self._linked = linked
        centers_label = fcluster(linked, t = K, criterion='maxclust')-1
        self.labels = np.zeros(len(labels_sb))
        for i, label in enumerate(labels_sb):
            self.labels[i] = centers_label[label]
        self.merge_cost = linked[:, 2][::-1]
        print("merge  end.")
    def show_merge(self):
        dendrogram(self._linked, orientation='top', labels=range(len(self.centers)), distance_sort='descending', show_leaf_counts=True)
        plt.xlabel('Prototype Index')
        plt.ylabel('Distance')
        plt.tight_layout()
        plt.show()
    def cost_curve(self):
        plt.plot(np.arange(1, len(self.centers)), self.merge_cost, color='#1f77b4', marker='o')
        plt.xticks(np.arange(1, len(self.centers)))
        plt.xlabel('Number of clusters')
        plt.ylabel('Cost')
        plt.tight_layout()
        plt.show()
    def show_prototype(self, lims=None):
        cluster_num = len(self.centers)
        rows = int(np.ceil(cluster_num / 5))
        fig = plt.figure(figsize=(30, 5*rows))
        if lims == None:
            for i in range(cluster_num):
                ax = fig.add_subplot(rows, 5, i+1)
                ax.plot(self.centers[i])
                ax.set_title(f"label={i}")
            plt.show()
        else:
            for i in range(cluster_num):
                ax = fig.add_subplot(rows, 5, i+1)
                ax.plot(self.centers[i]*(lims[0] - lims[1]) + lims[1])
                ax.set_title(f"label={i}")
            plt.show()
    def _density_gap(self, label_pred, K):
        mean_dist = np.mean(self.dist[self.dist!=0])
        delta = np.zeros(K)
        n_win = np.zeros(K)
        for i in range(K):
            c_idx = np.where(label_pred == i)[0]
            n_win[i] = len(c_idx)
            min_dist = []
            mean_density = np.mean(self.local_density[label_pred == i])
            for id in c_idx:
                density = self.local_density[id] 
                if density > mean_density:
                    larger_idx = np.where((self.local_density > density) & (label_pred == i))[0] 
                    if len(larger_idx) == 0: 
                        min_dist.append(0)
                    else:
                        t = np.min([self.dist[id][j] for j in larger_idx])
                        min_dist.append(t)
                else:
                    min_dist.append(0)
            delta[i] = np.max(min_dist) / mean_dist
        return np.argmax(delta * n_win)
    def _predict(self):
        d = pairwise_distances(self.centers, self.data) 
        c_pred = np.argmin(d, axis=0) 
        return c_pred