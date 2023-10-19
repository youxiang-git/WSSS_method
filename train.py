import torch
import os
import csv
from torch import optim
from dataloader import CustomVOCDataset
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights
from torchvision.models.segmentation import fcn_resnet50
from eval import model_eval

train = CustomVOCDataset("VOCtrainval_11-May-2012", "train", True)
val = CustomVOCDataset("VOCtrainval_11-May-2012", "val", False)

train_loader = DataLoader(train, batch_size=16, shuffle=True, num_workers=12)
val_loader = DataLoader(val, batch_size=16, shuffle=True, num_workers=12)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wsss_model = fcn_resnet50(weights=None, weights_backbone=ResNet50_Weights.IMAGENET1K_V1 , num_classes=21)
wsss_model.to(device)

optimizer = optim.Adam(wsss_model.parameters(), 1e-4)
# optimizer = optim.SGD(wsss_model.parameters(), lr=1e-4, momentum=0.9, weight_decay=0)
num_epochs = 1000
criterion = torch.nn.CrossEntropyLoss(ignore_index=255).to(device)

batches = len(train_loader)

best_miou = 0.0

for epoch in range(num_epochs):
    wsss_model.train()
    total_loss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):
        # if (batch_idx + 1) % 30 == 0:
        #     print(f"Now on batch - {batch_idx+1} / {batches}")
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = wsss_model(data)
        loss = criterion(output["out"], target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    loss_epoch = total_loss / batches

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss_epoch}")

    # if (epoch + 1) % 3 == 0:
    mean_iou = model_eval(wsss_model, val_loader, 21, device)
    print("Mean IoU on evaluation data: {}".format(mean_iou))
    with open("model_train_log.csv", "a") as file:
        writer = csv.writer(file)
        writer.writerow([epoch + 1, loss_epoch, mean_iou.item()])
        if mean_iou > best_miou:
            best_miou = mean_iou
            torch.save(
                wsss_model.state_dict(),
                "model_weights/voc_model_epoch_{}_weights.pth".format(epoch + 1),
            )
