import torch
import os
import csv
import numpy as np
from torchvision.datasets import VOCSegmentation
from torch import optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode
from torchvision.models.segmentation import fcn_resnet101

root = os.getcwd()

voc_root = os.path.join(root, "VOCtrainval_11-May-2012")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

transform = transforms.Compose(
    [
        transforms.Resize((256, 256), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ]
)
m_transform = transforms.Compose(
    [
        transforms.Resize((256, 256), interpolation=InterpolationMode.NEAREST),
        transforms.PILToTensor(),
    ]
)

train_dataset = VOCSegmentation(
    voc_root,
    year="2012",
    image_set="train",
    transform=transform,
    target_transform=m_transform,
)

val_dataset = VOCSegmentation(
    voc_root,
    year="2012",
    image_set="val",
    transform=transform,
    target_transform=m_transform,
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

wsss_model = fcn_resnet101(
        weights=None, weights_backbone=None, num_classes=21)

wsss_model.to(device)

optimizer = optim.AdamW(wsss_model.parameters())

num_epochs = 100

criterion = torch.nn.CrossEntropyLoss(reduction="mean")

batches = len(train_loader)

best_miou = 0.0

for epoch in range(num_epochs):
    wsss_model.train()
    total_loss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):
        if (batch_idx + 1) % 100 == 0:
            print(f"Now on batch - {batch_idx+1} / {batches}")
        data, target = data.to(device), target.to(device)

        print(target.shape)

        optimizer.zero_grad()
        output = wsss_model(data)
        loss = criterion(output["out"], target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    loss_epoch = total_loss / batches

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss_epoch}")

    if (epoch + 1) % 3 == 0:
        mean_iou = eval.model_eval(wsss_model, val_loader, 21, device)
        print("Mean IoU on evaluation data: {}".format(mean_iou))
        with open("model_train_log.csv", "a") as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, loss_epoch, mean_iou.item()])
        # if mean_iou > best_miou:
        #     best_miou = mean_iou
        #     torch.save(
        #         wsss_model.state_dict(),
        #         "model_weights/wsss_model_epoch_{}_weights.pth".format(epoch + 1),
        #     )

