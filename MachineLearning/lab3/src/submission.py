import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from safetensors.numpy import save_file, load_file
from pathlib import Path
from typing import Union
from PIL import Image
import json


# 1
class GMM:
    """
    Gaussian Mixture Model. Initialised by K-means clustering.

    We take X as the input data, which is a matrix of shape (N, D), where N is the number of samples and D is the dimension of the data.

    Args:
        - n_components: int, Number of components in the mixture model. Marked as K.
        - data_dim: int, Dimension of the data.Marked as D.

    Parameters:
        - means: np.ndarray, shape (K, D), Means of the Gaussian components.
        - covs: np.ndarray, shape (K, D, D), Covariances of the Gaussian components.
        - pi: np.ndarray, shape (K,), Mixing coefficients of the Gaussian components.

    Methods:
        - from_pretrained(path): Load the GMM model from a file.
        - fit(X, max_iter): Fit the GMM model to the data.
        - _e_step(X): E-step: Compute the responsibilities.
        - _m_step(X, gamma): M-step: Update the parameters.
        - _gaussian(X, mean, cov): Compute the Gaussian probability density function.
        - predict(X): Predict the cluster for each sample.
        - save_pretrained(path): Save the GMM model to a file.
    """

    def __init__(self, n_components: int, data_dim: int):
        self.n_components = n_components
        self.data_dim = data_dim
        self.means = np.random.rand(n_components, data_dim)
        self.covs = np.array([np.eye(data_dim) for _ in range(n_components)])
        self.pi = np.ones(n_components) / n_components

    @classmethod
    def from_pretrained(cls, path: Union[str, Path]):
        """
        Load the GMM model from a file.

        Args:
            - path: str, Path to load the model.
        """
        path = Path(path)
        with open(path / "config.json", "r") as f:
            config = json.load(f)
        params = load_file(path / "gmm.safetensors")
        model = cls(**config)
        model.means = params["means"]
        model.covs = params["covs"]
        model.pi = params["pi"]
        return model

    def fit(self, X: np.ndarray, max_iter: int = 100):
        """
        Fit the GMM model to the data.

        Args:
            - X: np.ndarray, shape (N, D), Data.
            - max_iter: int, Maximum number of iterations.
        """
        # Initialise the model with K-means
        kmeans = KMeans(n_clusters=self.n_components)
        kmeans.fit(X)
        self.means = kmeans.cluster_centers_
        self.covs = np.array([np.cov(X[kmeans.labels_ == i].T) for i in range(self.n_components)])
        self.pi = np.array([np.mean(kmeans.labels_ == i) for i in range(self.n_components)])

        # EM algorithm
        for _ in tqdm(range(max_iter)):
            # E-step
            gamma = self._e_step(X)
            # M-step
            self._m_step(X, gamma)

    def _e_step(self, X: np.ndarray) -> np.ndarray:
        """
        E-step: 计算后验概率
        Args:
            X: shape (N, D) 的数据矩阵
        Returns:
            gamma: shape (N, K) 的后验概率矩阵
        """
        N = X.shape[0]
        gamma = np.zeros((N, self.n_components))
        
        # 计算每个样本对每个高斯分量的响应度
        for k in range(self.n_components):
            inv_cov = np.linalg.inv(self.covs[k])
            det = np.linalg.det(self.covs[k])
            gamma[:, k] = self.pi[k] * self._gaussian(X, self.means[k], inv_cov, det)
        
        # 归一化确保每行和为1
        gamma = gamma / gamma.sum(axis=1, keepdims=True)
        return gamma

    def _m_step(self, X: np.ndarray, gamma: np.ndarray):
        """
        M-step: 更新模型参数
        Args:
            X: shape (N, D) 的数据矩阵
            gamma: shape (N, K) 的后验概率矩阵
        """
        N = X.shape[0]
        n_soft = np.sum(gamma, axis=0)  # [K,]
        
        # 更新混合系数 pi
        self.pi = n_soft / N
        
        # 更新均值
        self.means = (gamma.T @ X) / n_soft[:, np.newaxis]
        
        # 更新协方差矩阵
        for k in range(self.n_components):
            diff = X - self.means[k]  # shape: (N, D)
            # 修正：正确的矩阵乘法顺序
            self.covs[k] = (diff.T @ (gamma[:, k:k+1] * diff)) / n_soft[k]
            # 添加小的对角矩阵以确保数值稳定性
            self.covs[k] += 1e-6 * np.eye(self.data_dim)

    def _gaussian(self, X: np.ndarray, mean: np.ndarray, inv_cov: np.ndarray, det: float) -> np.ndarray:
        """
        Compute the Gaussian probability density function for a single component.

        Args:
            - X: np.ndarray, shape (N, D), Data.
            - mean: np.ndarray, shape (D,), Mean of the Gaussian component.
            - inv_cov: np.ndarray, shape (D, D), Inverse of the covariance matrix.
            - det: float, Determinant of the covariance matrix.

        Returns:
            - np.ndarray, shape (N,), Gaussian probability density function.
        """
        N, D = X.shape
        diff = X - mean
        exponent = np.sum(diff @ inv_cov * diff, axis=1)
        log_prob = -0.5 * exponent - 0.5 * np.log(det) - D / 2 * np.log(2 * np.pi)  # Prevent overflow
        return np.exp(log_prob)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the cluster for each sample.

        Returns:
            - np.ndarray, shape (N,), Predicted cluster for each sample.
        """
        gamma = self._e_step(X)
        return np.argmax(gamma, axis=1)

    def save_pretrained(self, path: Union[str, Path]):
        """
        Save the GMM model to a file.

        Args:
            - path: str, Path to save the model.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        config = {"n_components": self.n_components, "data_dim": self.data_dim}
        params = {"means": self.means, "covs": self.covs, "pi": self.pi}

        # Save config
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=4)

        # Save params
        save_file(params, path / "gmm.safetensors")


# 2
class PCA:
    """
    Principal Component Analysis.

    We take X as the input data, which is a matrix of shape (N, D), where N is the number of samples and D is the dimension of the data.

    Args:
        - dim: int, Number of components to keep. Marked as d.

    Parameters:
        - components: np.ndarray, shape (d, D), Principal components.
        - mean: np.ndarray, shape (D,), Mean of the data.

    Methods:
        - from_pretrained(path): Load the PCA model from a file.
        - fit(X): Fit the PCA model to the data.
        - transform(X): Project the data into the reduced space.
        - save_pretrained(path): Save the PCA model to a file.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.components = None
        self.mean = None

    @classmethod
    def from_pretrained(cls, path: Union[str, Path]):
        """
        Load the PCA model from a file.

        Args:
            - path: str, Path to load the model.
        """
        path = Path(path)
        with open(path / "config.json", "r") as f:
            config = json.load(f)
        params = load_file(path / "pca.safetensors")
        model = cls(**config)
        model.components = params["components"]
        model.mean = params["mean"]
        return model

    def fit(self, X: np.ndarray):
        """
        计算主成分
        Args:
            X: shape (N, D) 的数据矩阵
        """
        # 计算并保存均值
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # 计算协方差矩阵
        cov_matrix = np.cov(X_centered.T)
        
        # 计算特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # 选择最大的dim个特征值对应的特征向量
        idx = np.argsort(eigenvalues)[::-1][:self.dim]
        self.components = eigenvectors[:, idx].T  # shape (d, D)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project the data into the reduced space.

        Args:
            - X: np.ndarray, shape (N, D), Data.

        Returns:
            - X_pca: np.ndarray, shape (N, d), Projected data.
        """
        # Project data
        X = X - self.mean
        return X @ self.components.T

    def inverse_transform(self, X_pca: np.ndarray) -> np.ndarray:
        """
        Project the data back to the original space.

        Args:
            - X_pca: np.ndarray, shape (N, d), Projected data.

        Returns:
            - X: np.ndarray, shape (N, D), Original data.
        """
        return X_pca @ self.components + self.mean

    def save_pretrained(self, path: Union[str, Path]):
        """
        Save the PCA model to a file.

        Args:
            - path: str, Path to save the model.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        config = {"dim": self.dim}
        params = {"components": self.components, "mean": self.mean}

        # Save config
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=4)

        # Save params
        save_file(params, path / "pca.safetensors")


# 5
def sample_from_gmm(gmm: GMM, pca: PCA, label: int, path: Union[str, Path]):
    """
    从GMM中采样并生成图像
    Args:
        gmm: GMM模型
        pca: PCA模型
        label: 类别标签
        path: 保存路径
    """
    # 从指定类别的高斯分布中采样
    mean = gmm.means[label]
    cov = gmm.covs[label]
    sample = np.random.multivariate_normal(mean, cov, size=1)  # shape (1, d)
    
    # 使用PCA还原到原始维度
    sample_restored = pca.inverse_transform(sample)  # shape (1, D)
    
    # 将样本转换为图像格式 (28x28)
    image = sample_restored.reshape(28, 28)
    
    # 归一化到[0, 255]范围
    image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
    
    # 保存图像
    path = Path(path)
    Image.fromarray(image).save(path / "gmm_sample.png")
