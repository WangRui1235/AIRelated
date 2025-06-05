import argparse
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import utils
from utils import Model
import torch.optim as optim

def nt_xent_loss(out_1, out_2, temperature):
    """NT-Xent对比损失计算（支持梯度传播）"""
    # 使用支持梯度的all_gather
    out_1_gathered = torch.distributed.all_gather(out_1)
    out_2_gathered = torch.distributed.all_gather(out_2)
    
    # 拼接所有进程的输出
    out_1_all = torch.cat(out_1_gathered, dim=0)
    out_2_all = torch.cat(out_2_gathered, dim=0)
    
    total_batch_size = out_1_all.size(0)
    out = torch.cat([out_1_all, out_2_all], dim=0)
    
    # 计算相似度矩阵
    sim_matrix = torch.mm(out, out.t().contiguous()) / temperature
    
    # 排除自身相似度
    logits_mask = torch.eye(2 * total_batch_size, device=sim_matrix.device).bool()
    sim_matrix.masked_fill_(logits_mask, float('-inf'))
    
    # 正样本对
    pos_sim = torch.cat([
        torch.diag(sim_matrix[:total_batch_size, total_batch_size:]), 
        torch.diag(sim_matrix[total_batch_size:, :total_batch_size])
    ])
    
    # 计算损失
    loss = -pos_sim.mean() + torch.logsumexp(sim_matrix, dim=1).mean()
    return loss

def main_worker(rank, world_size, args):
    # 初始化分布式环境
    dist.init_process_group(backend='nccl', init_method='env://', 
                            world_size=world_size, rank=rank)
    device = torch.device(f'cuda:{rank}')
    
    # 数据加载
    train_data = utils.CIFAR10Pair(root='../data', train=True, 
                                 transform=utils.train_transform, download=True)
    train_sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, 
                             sampler=train_sampler, num_workers=16, 
                             pin_memory=True, drop_last=True)
    
    # 模型
    model = Model(feature_dim=128).to(device)
    model = DDP(model, device_ids=[rank])
    
    # 优化器
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)

    
    # 训练循环
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_sampler.set_epoch(epoch)
        
        for batch_idx, (pos_1, pos_2, _) in enumerate(train_loader):
            pos_1, pos_2 = pos_1.to(device), pos_2.to(device)
            
            # 前向
            _, out_1 = model(pos_1)
            _, out_2 = model(pos_2)
            
            # 计算损失
            loss = nt_xent_loss(out_1, out_2, args.temperature)
            
            # 反向
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # 保存模型（仅在主进程）
        if rank == 0:
            torch.save(model.module.state_dict(), 
                      f"model_epoch{epoch}.pth")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--optimizer', type=str, default='adam')
    args = parser.parse_args()
    
    # 自动获取分布式参数（通过torchrun启动）
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    main_worker(rank, world_size, args)