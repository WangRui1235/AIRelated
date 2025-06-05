import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA as SklearnPCA
import argparse
from utils import ae_encode
from model import AE
from datasets import load_from_disk
from submission import PCA, GMM
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", type=str, required=True)
    parser.add_argument("--cluster_label", action="store_true")
    args = parser.parse_args()

    save_path = Path(args.results_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # 加载数据和模型
    dataset = load_from_disk("../mnist_encoded")
    trainset = dataset["train"]
    traindata_raw2d = np.stack(trainset["image2D"])
    traindata_raw1d = np.stack(trainset["image1D"])
    true_labels = np.array(trainset["label"])

    # 加载训练好的GMM和PCA模型
    pca = PCA.from_pretrained(str(save_path / "pca"))
    gmm = GMM.from_pretrained(str(save_path / "gmm"))
    
    # 获取聚类标签
    traindata_pca = pca.transform(traindata_raw1d)
    cluster_labels = gmm.predict(traindata_pca)

    # 创建标签对应关系
    correspondence = np.zeros((10, 10))  # 10个真实标签 vs 10个聚类标签
    for true_label, cluster_label in zip(true_labels, cluster_labels):
        correspondence[true_label, cluster_label] += 1

    # 找出真实标签6最可能对应的聚类标签
    label_6_cluster = np.argmax(correspondence[6])
    print(f"\n真实标签6最可能对应的聚类标签是: {label_6_cluster}")
    print("\n对应关系矩阵:")
    print("行: 真实标签(0-9), 列: 聚类标签(0-9)")
    print(correspondence)
    
    # 计算每个真实标签对应的主要聚类标签
    for i in range(10):
        cluster = np.argmax(correspondence[i])
        total = np.sum(correspondence[i])
        accuracy = correspondence[i, cluster] / total * 100
        print(f"真实标签 {i} 主要对应聚类标签 {cluster} (准确率: {accuracy:.2f}%)")

    # 可视化
    plt.figure(figsize=(12, 5))
    
    # 真实标签的可视化
    plt.subplot(121)
    plt.scatter(traindata_pca[:, 0], traindata_pca[:, 1], 
               c=true_labels, cmap='tab10', alpha=0.5)
    plt.title("True Labels")
    plt.colorbar()
    
    # 聚类标签的可视化
    plt.subplot(122)
    plt.scatter(traindata_pca[:, 0], traindata_pca[:, 1], 
               c=cluster_labels, cmap='tab10', alpha=0.5)
    plt.title("Cluster Labels")
    plt.colorbar()
    
    plt.savefig(save_path / "label_comparison.png")
    plt.close()

    # 加载预训练的AE模型
    mnist_ae = AE.from_pretrained("Rosykunai/mnist-ae")

    print("AE encoding ...")
    traindata_ae = []
    # 添加通道维度
    traindata_raw2d = traindata_raw2d[:, np.newaxis, :, :]  # 形状变为(N, 1, 28, 28)
    
    # 分批处理以避免内存溢出
    batch_size = 1000
    for i in tqdm(range(0, len(traindata_raw2d), batch_size), desc="AE Encoding"):
        batch = traindata_raw2d[i:i+batch_size]
        traindata_ae.append(ae_encode(mnist_ae, batch))
    traindata_ae = np.concatenate(traindata_ae)
    
    print("tSNE fitting ...")
    tsne = TSNE(n_components=2)
    with tqdm(total=100, desc="t-SNE") as pbar:
        def update_progress(*args):
            pbar.update(1)
        traindata_tsne = tsne.fit_transform(traindata_ae, callbacks=update_progress)
    
    # 保存t-SNE结果的可视化
    plt.figure(figsize=(10, 10))
    plt.scatter(traindata_tsne[:, 0], traindata_tsne[:, 1], 
               c=true_labels, cmap='tab10', alpha=0.5)
    plt.title("t-SNE visualization")
    plt.colorbar()
    plt.savefig(save_path / "true_tsne.png")
    plt.close()
    
    print("PCA fitting ...")
    pca_vis = SklearnPCA(n_components=2)
    with tqdm(total=1, desc="PCA") as pbar:
        traindata_pca_vis = pca_vis.fit_transform(traindata_ae)
        pbar.update(1)
    
    # 保存PCA结果的可视化
    plt.figure(figsize=(10, 10))
    plt.scatter(traindata_pca_vis[:, 0], traindata_pca_vis[:, 1], 
               c=true_labels, cmap='tab10', alpha=0.5)
    plt.title("PCA visualization")
    plt.colorbar()
    plt.savefig(save_path / "true_pca.png")
    plt.close()
    
    # 保存AE结果的可视化
    plt.figure(figsize=(10, 10))
    plt.scatter(traindata_ae[:, 0], traindata_ae[:, 1], 
               c=true_labels, cmap='tab10', alpha=0.5)
    plt.title("AE visualization")
    plt.colorbar()
    plt.savefig(save_path / "true_ae.png")
    plt.close()

if __name__ == "__main__":
    main()
