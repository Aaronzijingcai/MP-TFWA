from torch import nn

class CELoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.xent_loss = nn.CrossEntropyLoss()

    def forward(self, predicts, targets):
        return self.xent_loss(predicts, targets)

