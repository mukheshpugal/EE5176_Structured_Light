import os
import torch
from model import networks
import numpy as np
from sys import argv
import cv2

def infer(net, lcn_in, image_in: np.ndarray):

    im = torch.tensor(image_in.reshape([1, 1, 480, 640])).to('cuda:0')

    im_lcn, im_std = lcn_in(im)
    im_cat = torch.cat((im_lcn, im), dim=1)
    
    return net(im_cat)[0][0].cpu().detach().numpy().reshape([480, 640])

def toImage(x):
    x = x.copy()

    x -= np.min(x)
    x /= np.max(x)
    return (255 * x).astype(np.uint8)

if __name__ == '__main__':

    imsizes = [(480,640)]
    for iter in range(3):
        imsizes.append((int(imsizes[-1][0]/2), int(imsizes[-1][1]/2)))

    net = networks.DispEdgeDecoders(channels_in=2, max_disp=128, imsizes=imsizes, output_ms=True)
    net.load_state_dict(torch.load('./net_0099.params'))

    net = net.to('cuda:0')
    net.eval()

    lcn_in = networks.LCN(5, 0.05)
    lcn_in = lcn_in.to('cuda:0')

    image_in = np.load(argv[-1])

    out = infer(net, lcn_in, image_in)
    image_out = toImage(out)

    cv2.imshow('disp', image_out)
    cv2.waitKey(0)
    