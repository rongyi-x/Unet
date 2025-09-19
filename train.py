#codeing=utf-8
import os

import torchvision

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import get_config
from model import Unet
from dataset import Carvana


def get_loaders(config):

    # transform 图像增强操作
    train_transform = A.Compose(
        [
            A.Resize(config['height'], config['width']),  # 图片大小调整为 512 x 512
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(mean=(0.0, 0.0, 0.0), std=(1, 1, 1), max_pixel_value=255.0),
            ToTensorV2()
        ]
    )

    valid_transform = A.Compose(
        [
            A.Resize(config['height'], config['width']),  # 图片大小调整为 512 x 512
            A.Normalize(mean=(0.0, 0.0, 0.0), std=(1, 1, 1), max_pixel_value=255.0),
            ToTensorV2()
        ]
    )

    train_dataset = Carvana(config['train_images'], config['train_masks'], transform=train_transform)

    valid_dataset = Carvana(config['valid_images'], config['valid_masks'], transform=valid_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,
                                  pin_memory=config['pin_memory'])

    valid_dataloader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True,
                                  pin_memory=config['pin_memory'])

    print(f'训练集大小: {len(train_dataset)}, 验证集大小: {len(valid_dataset)}')

    return train_dataloader, valid_dataloader


def train_model(config):
    # 准备工作
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用 {device} 进行训练')
    os.makedirs(config['model_folder'], exist_ok=True)
    writer = SummaryWriter(config['loss'])  # tensorboard --logdir=loss
    scaler = torch.cuda.amp.GradScaler()  # 自动混合精度 减少显存

    # 获取模型 数据集
    model = Unet().to(device)
    train_dataloader, valid_dataloader = get_loaders(config)

    # 损失函数优化器
    loss_fn = nn.BCEWithLogitsLoss().to(device)   # 二分类损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # # 动态学习率
    # scheduler = OneCycleLR(optimizer, max_lr=config['max_lr'], epochs=config['epoch'],
    #                        steps_per_epoch=len(train_dataloader))

    # 预加载训练模型
    model_path = config['model_folder']+config['preload_model']
    if os.path.exists(model_path):
        print(f"加载预训练模型{model_path}")
        state = torch.load(model_path)
        initial_epoch = state['epoch'] + 1
        global_step = state['global_step']
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        # scheduler.load_state_dict(state['scheduler_state_dict'])
    else:
        print(f"未找到预训练模型, 从头开始训练。")
        initial_epoch = 1
        global_step = 0

    # 训练
    for epoch in range(initial_epoch, config['epoch']+1):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch:02d}')

        for (data, target) in batch_iterator:
            data = data.to(device)

            # (batch_size, height, width) -> (batch_size, 1, height, width)
            target = target.unsqueeze(1).float().to(device)

            with torch.cuda.amp.autocast():
                predict = model(data)
                loss = loss_fn(predict, target)

            # 三件套 -> 五件套
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()  # 更新缩放因子
            # scheduler.step()  # 更新学习率

            # 后处理
            batch_iterator.set_postfix({"loss:": f"{loss.item():6.3f}"})
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()
            global_step += 1

        # 训练过程可视化
        visualization(model, valid_dataloader, device, epoch, config)

        # 保存模型
        print(f'保存模型: {model_path}')
        torch.save({
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            # "scheduler_state_dict": scheduler.state_dict()
        }, model_path)

        # 评估模型
        evalation(model, valid_dataloader, device)


def visualization(model, valid_dataloader, device, epoch, config, num=1):
    model.eval()
    count = 0
    print(f"\nepoch: {epoch} 正在预测图像... ")
    os.makedirs(config['result'], exist_ok=True)

    with torch.no_grad():
        for (x, y) in valid_dataloader:
            count += 1
            x = x.to(device)
            y = y.to(device).unsqueeze(1).float()
            pred = torch.sigmoid(model(x))  # (0, 1)
            pred = (pred > 0.5).float()  # 转化为二值图

            torchvision.utils.save_image(y, config['result'] + f'src_img_{epoch}_{count}.jpg')
            torchvision.utils.save_image(pred, config['result'] + f'pre_img_{epoch}_{count}.jpg')

            if count == num:
                break


def evalation(model, valid_dataloader, device):
    num_correct = 0
    num_pixels = 0
    dice_score = 0

    model.eval()
    with torch.no_grad():
        for x, y in valid_dataloader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            pred = torch.sigmoid(model(x))
            pred = (pred > 0.5).float()

            num_correct += (pred == y).sum()  # pred 和 y中相同像素点的个数
            num_pixels += torch.numel(pred)  # 统计 pred 中像素点的个数
            dice_score += (2 * (pred * y).sum()) / ((pred + y).sum()+1e-9)

    print(f'Got {num_correct}/{num_pixels} with accuracy {num_correct/num_pixels*100:.2f}%')

    # 相似度 Dice = 2*TP/(2*TP+FP+FN)
    print(f'Dice score : {dice_score / len(valid_dataloader)}')


if __name__ == '__main__':
    config = get_config()
    train_model(config)


