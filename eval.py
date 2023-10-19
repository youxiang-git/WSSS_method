import numpy as np
import torch
from torch.utils.data import DataLoader
from dataloader import CustomVOCDataset
from torchmetrics.classification import MulticlassJaccardIndex
from torchvision.models.segmentation import fcn_resnet50


def model_eval(model, dataloader, num_classes, device):
    model.eval()
    iou_metric = MulticlassJaccardIndex(num_classes, "macro", ignore_index=255).to(
        device
    )
    total_batches = len(dataloader)
    total_iou = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            # print(f"Now on batch - {batch_idx+1} / {total_batches} for evaluation")
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)["out"]
            _, preds = torch.max(outputs, 1)

            # print(preds_.shape)
            # np.save("preds", preds.squeeze(0).detach().cpu().numpy())

            # np.save("labels", labels.detach().cpu().numpy())
            preds_ = torch.reshape(preds, (-1,))
            labels_ = torch.reshape(labels, (-1,))
            # print(labels_.shape)
            # print(preds_[preds_ != 0])
            # print(labels_[labels_ != 0])

            iou = iou_metric(preds_, labels_)
            total_iou += iou * inputs.size(0)

            # print(iou)
            # print(inputs.size(0))
            total_samples += inputs.size(0)

    mean_iou = total_iou / total_samples
    return mean_iou


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wsss_model = fcn_resnet101(
        weights=None, weights_backbone=None, num_classes=21, progress=True
    )

    wsss_model.load_state_dict(
        torch.load("model_weights/wsss_model_epoch_48_weights.pth")
    )

    wsss_model.to(device)

    val = CustomVOCDataset("VOCtrainval_11-May-2012", "val", False)
    val_loader = DataLoader(val, batch_size=16, shuffle=True, num_workers=12)

    mean_iou = model_eval(wsss_model, val_loader, 21, device)
    print("Mean IoU on evaluation data: {}".format(mean_iou))
