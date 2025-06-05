import matplotlib.pyplot as plt
import re

# ---------------------- 数据解析函数 ----------------------
def parse_logs(content):
    # 初始化存储容器
    phase1_loss = []
    phase2_train_loss, phase2_train_acc = [], []
    phase2_test_loss, phase2_test_acc = [], []
    
    # 正则表达式模式
    pattern_phase1 = r"Train Epoch: (\d+)/300 Loss: (\d+\.\d+)"
    pattern_phase2_train = r"Train Epoch: \[(\d+)/250\] Loss: (\d+\.\d+) ACC: (\d+\.\d+)%"
    pattern_phase2_test = r"Test Epoch: \[(\d+)/250\] Loss: (\d+\.\d+) ACC: (\d+\.\d+)%"
    
    # 解析内容
    for line in content.split('\n'):
        # Phase 1: 300 Epochs
        if "300 Loss: " in line and "ACC" not in line:
            match = re.search(pattern_phase1, line)
            if match:
                epoch = int(match.group(1))
                loss = float(match.group(2))
                phase1_loss.append((epoch, loss))
        
        # Phase 2: 250 Epochs (Train)
        elif "Train Epoch: [" in line:
            match = re.search(pattern_phase2_train, line)
            if match:
                epoch = int(match.group(1))
                loss = float(match.group(2))
                acc = float(match.group(3))
                phase2_train_loss.append((epoch, loss))
                phase2_train_acc.append((epoch, acc))
        
        # Phase 2: 250 Epochs (Test)
        elif "Test Epoch: [" in line:
            match = re.search(pattern_phase2_test, line)
            if match:
                epoch = int(match.group(1))
                loss = float(match.group(2))
                acc = float(match.group(3))
                phase2_test_loss.append((epoch, loss))
                phase2_test_acc.append((epoch, acc))
    
    return phase1_loss, phase2_train_loss, phase2_train_acc, phase2_test_loss, phase2_test_acc

# ---------------------- 数据可视化函数 ----------------------
def plot_training_curves(phase1_loss, phase2_data):
    plt.figure(figsize=(15, 10))
    
    # Phase 1: 仅训练损失
    epochs_p1 = [x[0] for x in phase1_loss]
    loss_p1 = [x[1] for x in phase1_loss]
    plt.subplot(2, 2, 1)
    plt.plot(epochs_p1, loss_p1, 'b-', label='Train Loss')
    plt.title("Phase 1: Training Loss (300 Epochs)")
    plt.xlabel("Epoch"), plt.ylabel("Loss")
    plt.grid(True), plt.legend()
    
    # Phase 2: 训练/测试损失对比
    epochs_p2 = [x[0] for x in phase2_data[0]]
    train_loss = [x[1] for x in phase2_data[0]]
    test_loss = [x[1] for x in phase2_data[2]]
    plt.subplot(2, 2, 2)
    plt.plot(epochs_p2, train_loss, 'g-', label='Train Loss')
    plt.plot(epochs_p2, test_loss, 'r--', label='Test Loss')
    plt.title("Phase 2: Loss Comparison (250 Epochs)")
    plt.xlabel("Epoch"), plt.ylabel("Loss")
    plt.grid(True), plt.legend()
    
    # Phase 2: 训练/测试准确率对比
    train_acc = [x[1] for x in phase2_data[1]]
    test_acc = [x[1] for x in phase2_data[3]]
    plt.subplot(2, 2, 3)
    plt.plot(epochs_p2, train_acc, 'c-', label='Train Accuracy')
    plt.plot(epochs_p2, test_acc, 'm--', label='Test Accuracy')
    plt.title("Phase 2: Accuracy Comparison (250 Epochs)")
    plt.xlabel("Epoch"), plt.ylabel("Accuracy (%)")
    plt.grid(True), plt.legend()
    
    plt.tight_layout()
    plt.show()
    plt.savefig('traing_curves.png')

# ---------------------- 主程序 ----------------------
if __name__ == "__main__":
    # 假设文件内容已读入content变量
    with open("result.txt", "r") as f:
        content = f.read()
    
    # 解析数据
    phase1_loss, phase2_train_loss, phase2_train_acc, phase2_test_loss, phase2_test_acc = parse_logs(content)
    
    # 可视化
    plot_training_curves(phase1_loss, (phase2_train_loss, phase2_train_acc, phase2_test_loss, phase2_test_acc))