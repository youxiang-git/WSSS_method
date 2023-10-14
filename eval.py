import numpy as np
import torch
import os
from utils import CustomCOCODataset
from PIL import Image
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.transforms import InterpolationMode
from torchvision.models.segmentation import fcn_resnet50
from torchmetrics.classification import MulticlassJaccardIndex

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wsss_model = fcn_resnet50(
    weights=None, weights_backbone=None, num_classes=91, progress=True
)

img_transform = transforms.Compose(
    [
        transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ]
)

mask_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((224, 224), interpolation=InterpolationMode.NEAREST),
    ]
)

wsss_model.load_state_dict(torch.load("wsss_model_weights.pth"))

wsss_model.to(device)

mini_train_dataset = CustomCOCODataset(
    im_transform=img_transform, m_transform=mask_transform
)

eval_dataloader = DataLoader(
    mini_train_dataset, batch_size=1, shuffle=True, num_workers=0
)

iou_metric = MulticlassJaccardIndex(91, 'macro').to(device)

def model_eval(model, dataloader, num_classes):
    model.eval()

    total_iou = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)['out']
            _, preds = torch.max(outputs, 1)

            np.savetxt("labels.txt", labels.squeeze(0).detach().cpu().numpy(), fmt="%i")
            np.savetxt("preds.txt", preds.squeeze(0).detach().cpu().numpy(), fmt='%i')

            iou = iou_metric(preds, labels)
            print(iou)
            break
            total_iou += iou * inputs.size(0)
            total_samples += inputs.size(0)

    mean_iou = total_iou / total_samples
    return mean_iou


def compute_iou(pred, label, num_classes):
    pred = pred.view(-1)
    label = label.view(-1)

    iou_list = []
    for sem_class in range(num_classes):
        intersection = torch.sum((pred == sem_class) & (label == sem_class))
        union = torch.sum((pred == sem_class) | (label == sem_class))
        if union == 0:
            iou_list.append(
                float("nan")
            )  # If there is no ground truth, do not include in evaluation
        else:
            iou_list.append(float(intersection) / float(union))
    return sum(iou_list) / len(iou_list)


mean_iou = model_eval(wsss_model, eval_dataloader, 91)

print("Mean IoU on evaluation data: {}".format(mean_iou))
