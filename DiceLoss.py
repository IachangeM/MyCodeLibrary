import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target, loss_type='jaccard'):
        smooth = 1e-5
        inse = torch.sum(input * target)

        if loss_type == 'jaccard':
            l = torch.sum(input * input)
            r = torch.sum(target * target)
        elif loss_type == 'sorensen':
            l = torch.sum(input)
            r = torch.sum(target)
        else:
            raise Exception("Unknow loss_type")

        dice = (2. * inse + smooth) / (l + r + smooth)
        dice_loss = 1 - torch.mean(dice)
        return dice_loss


if __name__ == '__main__':
    loss_function = DiceLoss()
    x = torch.rand((1, 1, 32, 32))
    y = (torch.randn(1, 1, 32, 32) > 0.5).type(torch.FloatTensor)
    loss = loss_function(x, y)
    print(loss)

