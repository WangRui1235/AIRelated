import argparse
import os
import pandas as pd
import torch
import torch.optim as optim
from thop import profile, clever_format
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


import utils
from utils import Model


#Layer-wise Adaptive Rate Scaling，在源代码中使用了，
# 通过为每一层自适应地调整学习率来解决训练深层网络时，
# 某些层上出现梯度消失或爆炸的问题

class LARS(optim.Optimizer):  
    """LARS优化器实现（层自适应学习率缩放）"""  
    def __init__(self, params, lr=1.0, momentum=0.9):  
        """初始化参数"""  
        defaults = dict(lr=lr, momentum=momentum)  
        super(LARS, self).__init__(params, defaults)
    
    def __setstate__(self, state):  
        super(LARS, self).__setstate__(state)
    
    def step(self, closure=None):  
        """单步优化逻辑"""  
        loss = None  
        if closure is not None:  
            loss = closure()
        
        for group in self.param_groups:   
            momentum = group['momentum']  
            for p in group['params']:  
                if p.grad is None:  
                    continue
                grad = p.grad.data   
                state = self.state[p]
                
                # 初始化动量缓冲区
                if len(state) == 0:  
                    state['momentum_buffer'] = torch.zeros_like(p.data)
                momentum_buffer = state['momentum_buffer']    
                
                # 计算参数范数和梯度范数
                weight_norm = torch.norm(p.data).item()
                grad_norm = torch.norm(grad).item()
                
                # 计算局部学习率（信任比率）
                if weight_norm != 0 and grad_norm != 0:  
                    local_lr = 0.001 * weight_norm / (grad_norm + 1e-4 * weight_norm)
                else:  
                    local_lr = 1.0
                '''
                动量用于累积过去梯度的信息，帮助优化过程更平稳地收敛：
                            if self.classic_momentum:
                trust_ratio = 1.0
                if self._do_layer_adaptation(param_name):
                w_norm = tf.norm(param, ord=2)
                g_norm = tf.norm(grad, ord=2)
                trust_ratio = tf.where(
                    tf.greater(w_norm, 0), tf.where(
                        tf.greater(g_norm, 0), (self.eeta * w_norm / g_norm),
                        1.0),
                    1.0)
                scaled_lr = self.learning_rate * trust_ratio

                next_v = tf.multiply(self.momentum, v) + scaled_lr * grad
                if self.use_nesterov:
                update = tf.multiply(self.momentum, next_v) + scaled_lr * grad
                else:
                update = next_v
                next_param = param - update

                '''
                # 更新动量缓冲区
                momentum_buffer.mul_(momentum).add_(grad, alpha=local_lr * group['lr'])
                # 应用参数更新
                p.data.add_(momentum_buffer, alpha=-1.0)  
                # 应用权重衰减
                p.data.add_(p.data, alpha=-1*1e-4*group['lr'])  
        
        return loss


def nt_xent_loss(out_1, out_2, temperature, batch_size):
    """NT-Xent对比损失计算"""  
    # 拼接正负样本表示 [2B, D]
    out = torch.cat([out_1, out_2], dim=0)
    # 计算相似度矩阵 [2B, 2B]
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
    
    # 构建掩码（排除对角线元素）
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
    sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)
    pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)  # 对称扩展正样本
    loss = (-torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    return loss

def train(net, data_loader, train_optimizer):
    """单轮训练函数"""  
    net.train()
    total_loss, total_num = 0.0, 0
    train_bar = tqdm(data_loader, desc="Training")
    
    for pos_1, pos_2, target in train_bar:
        # 数据预处理（移动至GPU）
        pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
        
        # 前向传播获取特征和投影头输出
        feature_1, out_1 = net(pos_1)
        feature_2, out_2 = net(pos_2)
        
        # 计算对比损失
        loss = nt_xent_loss(out_1, out_2, temperature, batch_size)
        
        # 反向传播与优化
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()
        
        # 统计信息更新
        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description(
            f'Train Epoch: {epoch}/{epochs} Loss: {total_loss/total_num:.4f}'
        )
    
    return total_loss / total_num




if __name__ == '__main__':
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='SimCLR Training Pipeline')
    parser.add_argument('--temperature', default=0.5, type=float, help='Softmax温度参数')
    parser.add_argument('--batch_size', default=512, type=int, help='批量大小')
    parser.add_argument('--epochs', default=500, type=int, help='训练轮数')
    args = parser.parse_args()
    
    # 参数解包
    temperature , batch_size, epochs= args.temperature,args.batch_size, args.epochs
    
    # 数据加载器构建
    train_data = utils.CIFAR10Pair(
        root='../data', 
        train=True, 
        transform=utils.train_transform, 
        download=False
    )
    train_loader = DataLoader(
        train_data, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=16, 
        pin_memory=True, 
        drop_last=True
    )    
    test_data = utils.CIFAR10Pair(
        root='../data', 
        train=False, 
        transform=utils.test_transform, 
        download=False
    )
    test_loader = DataLoader(
        test_data, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=16, 
        pin_memory=True
    )
    
    # 模型与优化器初始化
    model = Model(feature_dim=128).cuda()
    
    # 优化器选择（Adam/LARS）
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    # optimizer = LARS(model.parameters(), lr=1.0, momentum=0.9)  # 如需使用LARS
    for epoch in range(1, epochs + 1):
        # 单轮训练
        train_loss = train(model, train_loader, optimizer)
        # 保存最优模型

    torch.save(model.state_dict(), f'{temperature}_{batch_size}_{epochs}_model.pth')
    print(f'保存最优模型成功！')



'''
def test(net, memory_data_loader, test_data_loader):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, _, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature, out = net(data.cuda(non_blocking=True))
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, _, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature, out = net(data)

            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()

            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * k, c, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                     .format(epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))

    return total_top1 / total_num * 100, total_top5 / total_num * 100



在实际进行中，我使用了KNN评估投票对比学习的准确率(PS:在官方仓库里，使用了线性评估的手段：)
    elif FLAGS.train_mode == 'pretrain' and FLAGS.lineareval_while_pretraining:
      # When performing pretraining and linear evaluation together we do not
      # want information from linear eval flowing back into pretraining network
      # so we put a stop_gradient.
      supervised_head_outputs = self.supervised_head(
          tf.stop_gradient(supervised_head_inputs), training)
我这里参考了https://github.com/leftthomas/SimCLR 的实现，对相似度进行温度缩放并取指数，
增强高相似度样本的权重，创建 one-hot 编码表示 K 个邻居的类别，根据相似度权重对 one-hot 标签进行加权求和，
得到每个类别的得分：因为这段代码本身也消耗不少时间，在辅助确定训练超参数后既没有包括这段代码，也没有提出新的处理方式
'''