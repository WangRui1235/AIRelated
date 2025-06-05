import subprocess
import numpy as np
from pathlib import Path

def run_experiment(embedding_dim, max_iter, reg_coef):
    """运行一次实验"""
    # 修改submission.py中的正则化系数
    with open('submission.py', 'r') as f:
        code = f.read()
    code = code.replace('1e-6 * np.eye(self.data_dim)', 
                       f'{reg_coef} * np.eye(self.data_dim)')
    with open('submission.py', 'w') as f:
        f.write(code)
    
    print(f"\n{'='*50}")
    print(f"Testing parameters:")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Max iterations: {max_iter}")
    print(f"Regularization: {reg_coef}")
    print(f"{'='*50}\n")
    
    # 运行训练
    train_cmd = f"python train.py --use_pca --embedding_dim {embedding_dim} --max_iter {max_iter}"
    subprocess.run(train_cmd.split())
    
    # 获取最新的结果目录
    results_dir = Path("../results")
    latest_dir = max(results_dir.glob("*"), key=lambda x: x.stat().st_mtime)
    
    # 运行评分脚本
    grade_cmd = f"python grade.py --sample_index 8 --results_path {str(latest_dir)}"
    subprocess.run(grade_cmd.split())
    
    input("\nPress Enter to continue to next parameter combination...")

# 参数网格
params = {
    'embedding_dim': [30, 50, 70],
    'max_iter': [50, 100],
    'reg_coef': [1e-7, 1e-6]
}

# 网格搜索
for dim in params['embedding_dim']:
    for iter_num in params['max_iter']:
        for reg in params['reg_coef']:
            try:
                run_experiment(dim, iter_num, reg)
            except Exception as e:
                print(f"Error occurred: {e}")
                continue

print("\nParameter tuning completed!") 