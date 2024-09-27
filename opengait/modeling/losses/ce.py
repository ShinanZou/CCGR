import torch.nn.functional as F

from .base import BaseLoss

class CrossEntropyLoss(BaseLoss):
    def __init__(self, scale=2**4, label_smooth=True, eps=0.1, loss_term_weight=1.0, log_accuracy=False):
        super(CrossEntropyLoss, self).__init__(loss_term_weight)
        self.scale = scale
        self.label_smooth = label_smooth
        self.eps = eps
        self.log_accuracy = log_accuracy

    def forward(self, logits, labels):
        """
            logits: [n, c, p]
            labels: [n]
        """
        n, c, p = logits.size()
        logits = logits.float()
        labels = labels.unsqueeze(1)
        if self.label_smooth:
            loss = F.cross_entropy(
                logits*self.scale, labels.repeat(1, p), label_smoothing=self.eps)
        else:
            loss = F.cross_entropy(logits*self.scale, labels.repeat(1, p))
        self.info.update({'loss': loss.detach().clone()})
        if self.log_accuracy:
            pred = logits.argmax(dim=1)  # [n, p]
            accu = (pred == labels).float().mean()
            self.info.update({'accuracy': accu})
        return loss, self.info


class view_CrossEntropyLoss(BaseLoss):
    def __init__(self, loss_term_weight=1.0):
        super(view_CrossEntropyLoss, self).__init__(loss_term_weight)
        # self.loss = nn.CrossEntropyLoss()

    def forward(self, logits_view, labels):
        """
            logits: [n, c]
            labels: [n]
        """
        loss = F.cross_entropy(logits_view, labels)
        self.info.update({'loss': loss.detach().clone()})

        return loss, self.info