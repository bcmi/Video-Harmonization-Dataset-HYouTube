import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import time
import numpy as np
import math
import trilinear
import cv2
import random
import tridistribute



    


class TridistributeGeneraotrFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mask, input, output):

        dim = 33
        #count to zero or one ?
        #depend on the initialization of lut
        #lut_count = torch.ones(lut.size())
        t1 = time.time()
        #print(input.device)
        batch = input.size(0)

        torch.cuda.set_device(int((str(input.device))[-1]))
        size = torch.Size((batch, 3, dim, dim, dim))
        lut = torch.zeros(size, device=input.device)
        lut_count = torch.zeros(size, device=input.device)

        input = input.contiguous()
        output = output.contiguous()
        lut = lut.contiguous()
        mask = mask.contiguous()

        lut_count = lut_count.contiguous()

        #dim = lut.size()[-1]
        binsize = 1.000001 / (dim - 1)
        W = input.size()[-1]
        H = input.size()[-2]

        shift = dim ** 3
        t2 = time.time()
        #print(input[index:index+1, :, :, :], output[index:index+1, :, :, :])
        assert 1 == tridistribute.forward(
            mask,
            lut,
            lut_count,
            input,
            output,
            dim,
            shift,
            binsize,
            W,
            H,
            batch
        )
        #lut_count[:, 1, :, :, :] = lut_count[:, 2, :, :, :] = lut_count[:, 0, :, :, :]
        #print("in lut")
        #print(lut.sum(), lut_count.sum())
        assert 1 == tridistribute.divide(
            lut,
            lut_count,
            dim,
            shift,
            batch
        )
        #print(lut.sum(), lut_count.sum())
        #print("in lut")
        int_package = torch.IntTensor([dim, shift, W, H, batch])
        float_package = torch.FloatTensor([binsize])
        variables = [mask, lut_count, input, int_package, float_package]
        ctx.save_for_backward(*variables)
        #print(lut_count)
        t2 = time.time()
        #print("::",t2 - t1)

        return lut,  lut_count, output

    @staticmethod
    def divide(lut, lut_count):
        #print("divide here")
        dim = lut.size()[-1]
        shift = dim ** 3
        batch = lut.size()[0]
        assert 1 == tridistribute.divide(
            lut,
            lut_count,
            dim,
            shift,
            batch
        )
        return lut, lut_count


    @staticmethod
    def backward(ctx, lut_grad):
        mask, lut_count, input, int_package, float_package = ctx.saved_variables
        dim, shift, W, H, batch = int_package
        dim, shift, W, H, batch = int(dim), int(shift), int(W), int(H), int(batch)
        binsize = float(float_package[0])
        output_grad = lut_grad.new(input.size())

        assert 1 == tridistribute.backward(
            mask,
            input,
            output_grad,
            lut_count,
            lut_grad,
            dim,
            shift,
            binsize,
            W,
            H,
            batch
        )

        return None, output_grad, lut_grad



class TridistributeGeneraotr(torch.nn.Module):
    def __init__(self):
        super(TridistributeGeneraotr, self).__init__()

    def forward(self, mask, input, output):
        return TridistributeGeneraotrFunction.apply(mask, input, output)

    def divide(self, lut, lut_count):
        return TridistributeGeneraotrFunction.divide(lut, lut_count)



class TrilinearInterpolationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lut_count, lut, x, fix_threshold = 0.01, k_threshold = 1):
        x = x.contiguous()

        output = x.new(x.size())
        #output_eff = x.new(torch.Size([x.size()[0],x.size()[2], x.size()[3]]))
        dim = lut.size()[-1]
        shift = dim ** 3
        binsize = 1.000001 / (dim-1)
        W = x.size(2)
        H = x.size(3)
        batch = x.size(0)
        #fix_threshold = 0.01
        #k_threshold = 1

        assert 1 == trilinear.forward(lut_count,
                                      lut,
                                      x, 
                                      output,
                                      dim, 
                                      shift, 
                                      binsize, 
                                      W, 
                                      H, 
                                      batch)


        int_package = torch.IntTensor([dim, shift, W, H, batch])
        float_package = torch.FloatTensor([binsize])
        variables = [lut, x, int_package, float_package]
        
        ctx.save_for_backward(*variables)
        
        return lut, output
    
    @staticmethod
    def backward(ctx, lut_grad, x_grad):
        
        lut, x, int_package, float_package = ctx.saved_variables
        dim, shift, W, H, batch = int_package
        dim, shift, W, H, batch = int(dim), int(shift), int(W), int(H), int(batch)
        binsize = float(float_package[0])
            
        assert 1 == trilinear.backward(x, 
                                       x_grad, 
                                       lut_grad,
                                       dim, 
                                       shift, 
                                       binsize, 
                                       W, 
                                       H, 
                                       batch)
        return lut_grad, x_grad

    @staticmethod
    def count_map(lut_count, x):
        x = x.contiguous()
        lut_count = lut_count.contiguous()
        #size = torch.Size((batch, 3, dim, dim, dim))
        output_map = x.new(x.size())
        output_map = output_map - output_map
        #print(output_map)
        # output_eff = x.new(torch.Size([x.size()[0],x.size()[2], x.size()[3]]))
        dim = lut_count.size()[-1]
        shift = dim ** 3
        binsize = 1.000001 / (dim - 1)
        W = x.size(2)
        H = x.size(3)
        batch = x.size(0)

        assert 1 == trilinear.map_count(lut_count,
                                      x,
                                      output_map,
                                      dim,
                                      shift,
                                      binsize,
                                      W,
                                      H,
                                      batch)

        return output_map


class TrilinearInterpolation(torch.nn.Module):
    def __init__(self, fix_threshold=0.01, k_threshold=1):
        super(TrilinearInterpolation, self).__init__()
        self.fix_threshold = fix_threshold
        self.k_threshold = k_threshold

    def forward(self, lut_count, lut, x):
        return TrilinearInterpolationFunction.apply(lut_count, lut, x, self.fix_threshold, self.k_threshold)

    def count_map(self, lut_count, x):
        return TrilinearInterpolationFunction.count_map(lut_count, x)


