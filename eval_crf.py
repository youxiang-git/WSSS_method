import numpy as np
import torch
from torch.utils.data import DataLoader
from dataloader_crf import CustomCRFDataset
from torchmetrics.classification import (
    MulticlassJaccardIndex,
)
from PIL import Image
from torchvision.models.segmentation import fcn_resnet50
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from pydensecrf import densecrf
from pydensecrf.utils import create_pairwise_bilateral, unary_from_softmax


def crf_postprocess(probs, img, n_labels):
    _, h, w = probs.shape

    # The input should be negative log probabilities
    unary = unary_from_softmax(probs)

    # The inputs should be C-continious
    unary = np.ascontiguousarray(unary)

    d = densecrf.DenseCRF2D(w, h, n_labels)
    d.setUnaryEnergy(unary)

    # Potential penalties for edge-preserving appearance
    # You can adjust these parameters as needed
    d.addPairwiseGaussian(
        sxy=(3, 3),
        compat=3,
        kernel=densecrf.DIAG_KERNEL,
        normalization=densecrf.NORMALIZE_SYMMETRIC,
    )
    d.addPairwiseBilateral(
        sxy=(15, 15),
        srgb=(3, 3, 3),
        rgbim=img,
        compat=10,
        kernel=densecrf.DIAG_KERNEL,
        normalization=densecrf.NORMALIZE_SYMMETRIC,
    )

    # Inference
    q = d.inference(5)  # You can adjust the number of iterations
    predictions = np.argmax(q, axis=0).reshape((h, w))

    return predictions


def model_eval(model, dataloader, num_classes, device):
    model.eval()
    total_loss = 0.0
    iou_metric = MulticlassJaccardIndex(num_classes, "macro", ignore_index=255).to(
        device
    )
    criterion1 = torch.nn.CrossEntropyLoss(ignore_index=255).to(device)
    # criterion2 = DiceLoss(softmax=True, to_onehot_y=True).to(device)
    total_batches = len(dataloader)

    with torch.no_grad():
        for batch_idx, (inputs, labels, img) in enumerate(dataloader):
            if (batch_idx+1 % 100) == 0:
              print(f"Now on batch - {batch_idx+1} / {total_batches} for evaluation")
            inputs = inputs.to(device)
            labels = labels.to(device)
            # print(img.shape)
            img_pil = TF.to_pil_image(img.squeeze(0))
            img2pass = img_pil.resize([256, 256], Image.BILINEAR)
            img_np = np.array(img2pass)
            outputs = model(inputs)["out"]
            probabilities = F.softmax(outputs, dim=1)
            probabilities_np = probabilities.detach().cpu().numpy()
            probabilities_np = probabilities_np.squeeze(0)
            # print(probabilities_np.dtype)
            refined_predictions = crf_postprocess(probabilities_np, img_np, 21)
            refined_predictions_tensor = torch.from_numpy(refined_predictions).to(
                device
            )
            # _, preds = torch.max(outputs, 1)

            # print(outputs.shape)
            # print(preds.shape)
            # print(labels.shape)
            # np.save("preds", preds.squeeze(0).detach().cpu().numpy())

            # np.save("labels", labels.detach().cpu().numpy())
            preds_ = torch.reshape(refined_predictions_tensor, (-1,))
            labels_ = torch.reshape(labels, (-1,))
            # print(labels_.shape)
            # print(preds_[preds_ != 0])
            # print(labels_[labels_ != 0])
            loss = criterion1(outputs, labels)
            # labels[labels == 255] = 0
            # loss = loss1 + (2 * criterion2(outputs, labels.unsqueeze(1)))
            iou_metric.update(preds_, labels_)
            total_loss += loss.item()

            # mAP_metric.update(outputs, labels)
            # total_iou += iou * inputs.size(0)
            # total_map += mAP * inputs.size(0)

            # print(iou)
            # print(inputs.size(0))
            # total_samples += inputs.size(0)

        loss_epoch = total_loss / total_batches

    # mean_iou = total_iou / total_samples
    # mean_map = total_map / total_samples
    return iou_metric.compute(), loss_epoch


# , mAP_metric.compute()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wsss_model = fcn_resnet50(
        weights=None, weights_backbone=None, num_classes=21, progress=True
    )

    wsss_model.load_state_dict(torch.load("best_model/weak_ce_dice.pth"))

    wsss_model.to(device)

    val = CustomCRFDataset("VOCtrainval_11-May-2012", "val", False)
    val_loader = DataLoader(val, batch_size=1, shuffle=True, num_workers=12)

    mean_iou, _ = model_eval(wsss_model, val_loader, 21, device)
    print("Mean IoU on evaluation data: {}".format(mean_iou))
