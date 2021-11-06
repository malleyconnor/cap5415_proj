import torch.nn as nn

IMAGENET_RES = ()

class Reconstructor(nn.Module):
    def __init__(self, FLAGS):
        super(Reconstructor, self).__init__()

        self.Conv1 = nn.Conv2d()