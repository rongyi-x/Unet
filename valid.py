
import torch
from dataset import Carvana
from config import get_config
from torch.utils.data import DataLoader
import albumentations as ab
from albumentations.pytorch import ToTensorV2
from model import Unet
from train import visualization


config = get_config()
device = 'cuda'

test_transform = ab.Compose([
    ab.Resize(config['height'], config['width']),
    ab.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
    ToTensorV2()
])

test_dataset = Carvana(config['test_images'], config['test_masks'],
                       transform=test_transform)

test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True,
                              pin_memory=config['pin_memory'])

print(f"测试集大小: {len(test_dataset)}")

# 导入模型
model_path = config['model_folder']+config['preload_model']
state = torch.load(model_path)
model = Unet().to(device)
model.load_state_dict(state["model_state_dict"])

visualization(model, test_dataloader, device, 'test', config, 3)




