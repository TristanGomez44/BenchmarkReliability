
import torch 
import torch.nn.functional as F 
import numpy as np

def ece(logits, labels,n_bins=15):
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
   
    softmaxes = F.softmax(logits, dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    accuracies = predictions.eq(labels)

    ece = torch.zeros(1, device=logits.device)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |confidence - accuracy| in each bin
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece.item()

def histedges_equalN(x,nbins):
    npt = len(x)
    return np.interp(np.linspace(0, npt, nbins + 1),
                    np.arange(npt),
                    np.sort(x))

def ada_ece(logits, labels,nbins=15):

    softmaxes = F.softmax(logits, dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    accuracies = predictions.eq(labels)
    n, bin_boundaries = np.histogram(confidences.cpu().detach(), histedges_equalN(confidences.cpu().detach(),nbins))
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    ece = torch.zeros(1, device=logits.device)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |confidence - accuracy| in each bin
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece.item()

def class_ece(logits, labels,n_bins=15):

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    num_classes = int((torch.max(labels) + 1).item())
    softmaxes = F.softmax(logits, dim=1)
    per_class_sce = None

    for i in range(num_classes):
        class_confidences = softmaxes[:, i]
        class_sce = torch.zeros(1, device=logits.device)
        labels_in_class = labels.eq(i) # one-hot vector of all positions where the label belongs to the class i

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = class_confidences.gt(bin_lower.item()) * class_confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = labels_in_class[in_bin].float().mean()
                avg_confidence_in_bin = class_confidences[in_bin].mean()
                class_sce += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        if (i == 0):
            per_class_sce = class_sce
        else:
            per_class_sce = torch.cat((per_class_sce, class_sce), dim=0)

    sce = torch.mean(per_class_sce)
    return sce.item()