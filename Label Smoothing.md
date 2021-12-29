# label smoothing loss for classification
### get code from https://github.com/OpenNMT/OpenNMT-py/blob/e8622eb5c6117269bb3accd8eb6f66282b5e67d9/onmt/utils/loss.py

```Python
import torch.nn.functional as F
class LabelSmoothingLoss(nn.Module):
    def __init__(self, label_smoothing, classes, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__()
        smoothing_value = label_smoothing / (classes)  ### classes is the number of classes
        one_hot = torch.full((classes,), smoothing_value)
        #one_hot[self.ignore_index] = 0  
        self.register_buffer('one_hot', one_hot.unsqueeze(0))
        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

        return F.kl_div(output, model_prob, reduction='sum') 
```
### general code
    ...
    label_smoothing, classes = 0.1, 3
    criterion = LabelSmoothingLoss(label_smoothing, classes)
    logit = model(input) ## (batch, classes), generally obtained by fc layer
    log_soft_logit = F.log_softmax(logit, dim=-1) 
    loss = criterion(log_soft_logit, label)   ### the result is the sum of one batch, so can use loss/ batch_size to get loss of per example
    
    
    
