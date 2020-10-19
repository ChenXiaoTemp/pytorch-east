import torch
import torch.nn as nn


class GRUGate(nn.Module):
    def __init__(self,in_channels:int,out_channels:int,kernel_size1=1,kernel_size2=1,stride=1,padding=0,dilation=1,groups=1,bias=True,padding_mode:str="zeros"):
        self._w = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size1,
                                  stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False,
                                  padding_mode=padding_mode)
        self._u = nn.Conv2d(in_channels=in_channels,out_channels=out_channels, kernel_size=kernel_size2,
                                  stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
                                  padding_mode=padding_mode)
        self._activation = nn.Sigmoid()

    def forward(self,x):
        x = self._w(x)+self._u(x)
        x = self._activation(x)
        return x


class GRUCandidate(nn.Module):
    def __init__(self,in_channels:int,out_channels:int,kernel_size1=1,kernel_size2=1,stride=1,padding=0,dilation=1,groups=1,bias=True,padding_mode:str="zeros"):
        self._w = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size1,
                                  stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
                                  padding_mode=padding_mode)
        self._u = nn.Conv2d(in_channels=in_channels,out_channels=out_channels, kernel_size=kernel_size2,
                                  stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
                                  padding_mode=padding_mode)
        self._activation = nn.Tanh()

    def forward(self,x,r):
        x = self._w(x)+r*self._u(x)
        x = self._activation(x)
        return x


class GRULayer(nn.Module):

    def __init__(self,channels:int,kernel_size1=1,kernel_size2=1,stride=1,padding=0,dilation=1,groups=1,bias=True,padding_mode:str="zeros"):
        self._z=GRUGate(in_channels=channels,out_channels=channels,kernel_size1=kernel_size1,kernel_size2=kernel_size2,stride=stride,padding=padding,dilation=dilation,groups=groups,bias=bias,padding_mode=padding_mode)
        self._r=GRUGate(in_channels=channels,out_channels=channels,kernel_size1=kernel_size1,kernel_size2=kernel_size2,stride=stride,padding=padding,dilation=dilation,groups=groups,bias=bias,padding_mode=padding_mode)
        self._h1=GRUCandidate(in_channels=channels,out_channels=channels,kernel_size1=kernel_size1,kernel_size2=kernel_size2,stride=stride,padding=padding,dilation=dilation,groups=groups,bias=bias,padding_mode=padding_mode)


    def forward(self,x):
        z=self._z(x)
        r=self._r(x)
        h1=self._h1(x,r)
        h=(1-z)*x+z*h1
        return h



