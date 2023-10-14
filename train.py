import torch
import numpy as np
import os
from utils import CustomCOCODataset, img_transform, mask_transform
from PIL import Image
from torch import optim
from torch.utils.data import DataLoader, Dataset

from torchvision.models.segmentation import fcn_resnet50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



mini_train_dataset = CustomCOCODataset(
    im_transform=img_transform, m_transform=mask_transform
)

mini_train_loader = DataLoader(
    mini_train_dataset, batch_size=24, shuffle=True, num_workers=4
)

wsss_model = fcn_resnet50(
    weights=None, weights_backbone=None, num_classes=91, progress=True
)

wsss_model.to(device)

optimizer = optim.Adam(wsss_model.parameters(), lr=0.0001)

num_epochs = 30

criterion = torch.nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    wsss_model.train()
    total_loss = 0.0

    for batch_idx, (data, target) in enumerate(mini_train_loader):
        if (batch_idx + 1) % 100 == 0:
            print(f"Now on batch - {batch_idx+1}")
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = wsss_model(data)
        loss = criterion(output["out"], target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(mini_train_loader)}")

torch.save(wsss_model.state_dict(), "wsss_model_weights.pth")
