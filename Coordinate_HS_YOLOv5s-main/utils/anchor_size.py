import torch
import numpy as np

def calcAnchorSize(targets, thres=[.0033, .03]):
    anchor_size = targets[:,-2]*targets[:,-1]
    return torch.cat((anchor_size>=thres[0], anchor_size>=thres[1])).reshape(2,-1).sum(dim=0)


def sizeLabel(i, tbox):
    thres = [55*55, 166*166]
    tsize = (tbox[:,3] - tbox[:,1]) * (tbox[:,2] - tbox[:,0])
    size_cls = torch.cat((tsize>=thres[0], tsize>=thres[1])).reshape(2,-1).sum(dim=0)
    return size_cls[i]





if __name__ == '__main__':
    """test: calcAnchorSize"""
    # targets = torch.tensor([[0,1,2,3,166,225],
    #            [1,2,3,4,155,135],
    #            [1,2,2,4,33,24]])
    # 
    # print(calcAnchorSize(targets))


    """test: sizeLabel"""
    pred_cls = torch.tensor([1,1,2,3])
    tcls = torch.tensor([1,1,2,4])
    size_cls = torch.tensor([0,1,2,3])

    print(1)

    print(sizeLabel(pred_cls, tcls, size_cls))