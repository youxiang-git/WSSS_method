import csv
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision.models.segmentation import fcn_resnet50

import eval
from utils import CustomCOCODataset, img_transform, mask_transform


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mini_train_dataset = CustomCOCODataset(
        "coco_minitrain2017", "pseudo_labels", img_transform, mask_transform
    )

    mini_val_dataset = CustomCOCODataset(
        "val2017", "val2017_masks", img_transform, mask_transform
    )

    mini_train_loader = DataLoader(
        mini_train_dataset, batch_size=16, shuffle=True, num_workers=24
    )

    mini_val_loader = DataLoader(
        mini_val_dataset, batch_size=64, shuffle=True, num_workers=24
    )

    wsss_model = fcn_resnet50(
        weights=None, weights_backbone=None, num_classes=81, progress=True
    )

    wsss_model.to(device)

    optimizer = optim.AdamW(wsss_model.parameters())

    num_epochs = 100

    criterion = torch.nn.CrossEntropyLoss(reduction="mean")

    batches = len(mini_train_loader)

    best_miou = 0.0

    for epoch in range(num_epochs):
        wsss_model.train()
        total_loss = 0.0

        for batch_idx, (data, target) in enumerate(mini_train_loader):
            if (batch_idx + 1) % 100 == 0:
                print(f"Now on batch - {batch_idx+1} / {batches}")
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = wsss_model(data)
            loss = criterion(output["out"], target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        loss_epoch = total_loss / batches

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss_epoch}")

        if (epoch + 1) % 3 == 0:
            mean_iou = eval.model_eval(wsss_model, mini_val_loader, 81, device)
            print("Mean IoU on evaluation data: {}".format(mean_iou))
            with open("model_train_log.csv", "a") as file:
                writer = csv.writer(file)
                writer.writerow([epoch + 1, loss_epoch, mean_iou.item()])
            if mean_iou > best_miou:
                best_miou = mean_iou
                torch.save(
                    wsss_model.state_dict(),
                    "model_weights/wsss_model_epoch_{}_weights.pth".format(epoch + 1),
                )


if __name__ == "__main__":
    train()
