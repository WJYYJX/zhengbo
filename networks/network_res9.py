# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# from convlstm import ConvLSTM
#from utee import misc
# print = misc.logger.info
import torch
#import cv2
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math



model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def mack_conv(size1, size2):
    with torch.no_grad():
        a=torch.ones(1,1,size1*size2 ,size1*size2)
        for i  in range( size1*size2):
            a[0,0,i,i//size2*size2:i//size2*size2+size2 ]=0
    return a

# def conv3x3(in_planes, out_planes, stride=1):
#     """3x3 convolution with padding"""
#     return nn.Cov2d_v3(in_planes, out_planes, kernel_size=3, stride=stride,
#                      padding=1, bias=False)




class PreNorm2(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        # self.norm = nn.LayerNorm(dim)
        self.norm = nn.BatchNorm1d(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        a1,a2,a3,a4=x.shape
        x=x.permute(0,3,1,2).reshape(a1,a4,-1)
        x=self.norm(x)
        x=x.reshape( a1,a4,a2,a3).permute(0,2,3,1)
        return self.fn( x, **kwargs)
    

class FeedForward2(nn.Module):
    def __init__(self, dim, hidden_dim,heads = 8, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
    
    
    

    
    
class FeedForward3(nn.Module):
    def __init__(self, dim, hidden_dim,heads = 8, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim,1,groups=6),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, dim,1,groups=6),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
    
    

class Attention2(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.gelu=nn.GELU()

        self.to_qkv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.kc = nn.Linear(dim, inner_dim , bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(2, dim = -1)                                                                         #在最后一维分为两份
        q,  v = map(lambda t: rearrange(t, 'b n1 n2 (h d) -> b  n1 h n2  d', h= self.heads), qkv)                       #对分开后的张量进行计算
        
        k = self.kc(x-torch.mean(x,-1,keepdim=True))                                                                    #引导
        k=rearrange(k, 'b n1 n2 (h d) -> b  n1 h n2  d', h= self.heads)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b n1 h n2  d -> b n1  n2 (h d)')

        return self.to_out(out)

    
class trans_conv(nn.Module):
    def __init__(self, inplane, outplane, kernelsize=1,padding=0, stride=1, size=(4,4 )):
        super(trans_conv, self).__init__()
        self.regionsize=kernelsize                                                                                      #卷积核大小
        self.padding=padding                                                                                            #拓展
        # self.padding=0
        self.stride=stride                                                                                              #步长
        self.scale1=(size[0]+self.padding*2-self.regionsize)//self.stride+1                                             #图像w
        self.scale2=( size[1]+self.padding*2-self.regionsize)//self.stride+1                                            #图像h
        self.scale=self.scale1*self.scale2                                                                              #图像像素数
        self.scaleplus=self.scale+1                                                                                     #
        num_patches=size[0]*size[1]                                                                                     #patch数目
        self.inchannel=inplane
        self.outchannel=outplane
        depth=1                                                                                                         #维度=输出通道整除6
        dim=outplane//6
        self.d= 4
        heads=self.d*1                                                                                                  #注意力头
        dim_head=dim//heads                                                                                             #每个注意力头维度
        dropout=0.1
        # mlp_dim=2048//1
        mlp_dim=dim*2//1                                                                                                #mlp维度
        self.dim=dim
        # patch_height=size[0]//1
        patch_height=1
        # patch_width=size[1]//1
        patch_width=1
        self.resh=  Rearrange('b (c k) (h p1) (w p2) -> b  (k p1 p2) (h w) c',k=6, p1 = patch_height, p2 = patch_width) #合并维度  #手动切patch1*1
        self.resh2=  Rearrange('b (k p1 p2) (h w)  c -> b (c k) (h p1) (w p2)',k=6, p1 = patch_height, p2 = patch_width,h=math.ceil(size[0]/patch_height),w=math.ceil(size[1]/patch_width)  )#还原
       
        if size[0]%patch_height==0:
            self.pad1=0
        else:
            self.pad1=patch_height-math.ceil(size[0]%patch_height)
        if size[1]%patch_width==0:
            self.pad2=0
        else:
            self.pad2=patch_height-math.ceil(size[1]%patch_width)

        self.pos_embedding = nn.Parameter(torch.randn(1,6*patch_height*patch_width, num_patches , dim))



        self.fc0=nn.Conv2d(dim,dim, kernel_size=self.regionsize,padding=padding,groups=dim )                            #卷积
        

        self.conv1=nn.Conv2d( self.inchannel,dim,kernel_size=3,padding=1,stride=1)

        self.gelu=nn.GELU()

        self.norm0_1=nn.BatchNorm2d( dim )
        # self.ffc=FeedForward3(dim, mlp_dim, heads = heads, dropout = dropout)

        self.bn2=nn.BatchNorm2d(outplane)
        self.bn3=nn.BatchNorm2d(dim)

        self.layers = nn.ModuleList([])
        for _ in range(depth):                                                                                          #注意力机制
            self.layers.append(nn.ModuleList([
                PreNorm2(dim, Attention2(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm2(dim, FeedForward2(dim, mlp_dim, heads = heads, dropout = dropout))
            ]))

        self.dropout = nn.Dropout(dropout)
        self.to_latent = nn.Identity()
        self.norm3=nn.LayerNorm([dim])

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 10)
        )

    def forward(self, x  ):
        # x1=self.bn2(self.gelu(self.conv1(x)  )) 
        x1=x

        pad=[self.pad1,0,self.pad2,0]
        x1=F.pad(x1,pad,mode='constant',value=0)


        x1=self.resh(x1)                                                                                                #改变维度

        b1,b2,a1,a2=x1.shape                                                                                            # b  (k p1 p2) (h w) c

        x1 += self.pos_embedding[:,:, :(a1)]* (x1!=0).float()

        x1 = self.dropout(x1)

        for attn, ff in self.layers:
            x1 = attn(x1) +x1 
            x1 = ff(x1) + x1


        x1=self.resh2(x1)[:,:,self.pad1:,self.pad2:]                                                                    #b (c k) (h p1) (w p2)
        
        
        # x2=self.bn3(x1)
        # x1=self.ffc(x2)+x1
        
         
        
        

        
        # x1=self.norm0_1(self.gelu(self.fc0(x1)  )) +x1
        # x1=self.bn2(self.gelu(self.conv1(x)  )) +x1

        

        return x1












class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):                                                                                           #mlp
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.,mask=None):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        self.mask=mask

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)                              #qkv的shape [b,num_heads,patch_size,dim//num_heads]

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if self.mask is not None:

            dots=dots*self.mask

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)






def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1,groups=6, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1,groups=6, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        # self.bn1 = nn.BatchNorm2d(planes//6)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        # self.bn2 = nn.BatchNorm2d(planes//6)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        # a,b,c,d=out.shape
        # out=out.reshape(a,b//6 ,6  , c,d).reshape(a,b//6 ,6,-1 )
        # out = self.bn1(out).reshape(a,b//6,6,c,d  ).reshape( a,b,c,d)

        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # a,b,c,d=out.shape
        # out=out.reshape(a,b//6 ,6  , c,d).reshape(a,b//6 ,6,-1 )
        # out = self.bn2(out).reshape(a,b//6,6,c,d  ).reshape( a,b,c,d)


        # if self.downsample is not None:
        #     identity = self.downsample(x)
        
        if self.downsample is not None:
            identity = self.downsample(x)

            # for index0, layer in enumerate(self.downsample ):
            #     # print( (index0,layer ))
            #     if index0==0:
            #         x1=layer(x)
            #     else:
            #         a,b,c,d=x1.shape
            #         x1=x1.reshape(a,b//6 ,6  , c,d).reshape(a,b//6 ,6,-1 )
            #         identity = layer(x1).reshape(a,b//6,6,c,d  ).reshape( a,b,c,d)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

            # for index0, layer in enumerate(self.downsample ):
            #     # print( (index0,layer ))
            #     if index0==0:
            #         x1=layer(x)
            #     else:

            #         a,b,c,d=x1.shape
            #         x1=x1.reshape(a,b//6 ,6  , c,d).reshape(a,b//6 ,6,-1 )
            #         identity = layer(x1).reshape(a,b//6,6,c,d  ).reshape( a,b,c,d)




        out += identity
        out = self.relu(out)

        return out










def whitening(im):
    batch_size, channel, h, w = im.shape
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    im = torch.cat([(im[:,[0]]-mean[0])/std[0],
                    (im[:,[1]]-mean[1])/std[1],
                    (im[:,[2]]-mean[2])/std[2]], 1)
    return im

def l2_norm(x):
    norm = torch.norm(x, p=2, dim=1, keepdim=True)
    x = torch.div(x, norm)
    return x

class ResNet18(nn.Module):
    def __init__(self, model, num_classes=1000):
        super(ResNet18, self).__init__()
        self.backbone = model
        # self.backbone = model.backbone
        self.cc=64*6
        self.inplanes = self.cc
        ss=2
        self.mask=mack_conv(6,ss**2 ).cuda()
        # self.mask=None
        
        self.fc1 = nn.Linear(512, num_classes)

        depth=1
        dim=self.cc*8//6
        self.dim=dim
        heads=4
        dim_head=dim//heads
        dropout=0.1
        mlp_dim=self.cc*2//6

        self.fc0=nn.Linear( self.cc*8 //6,dim )

        self.norm0=nn.LayerNorm( [  self.cc*8 //6])
        self.norm0_1=nn.LayerNorm( [dim])
        self.pos_embedding = nn.Parameter(torch.randn(1, 6*4+1,   dim))
        # self.cls_token = nn.Parameter(torch.randn(1, 1, self.regionsize**2*self.inchannel)//self.regionsize**2)  
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) )  

        self.layers = nn.ModuleList([])                                                                                 #transformer层
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout,mask=self.mask)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

        self.dropout = nn.Dropout(dropout)
        self.to_latent = nn.Identity()

        # self.fc3=nn.Linear(self.regionsize**2,self.regionsize**2)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        self.gelu=nn.GELU()


        # self.conv0= nn.Conv2d(6, self.cc, kernel_size=7, stride=2, padding=3,groups=6, bias=False)
        # # self.bn0=nn.BatchNorm2d(self.cc//6)
        # self.bn0=nn.BatchNorm2d(self.cc)
        # self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2)

        # self.conv0= nn.Conv2d(6, self.cc, kernel_size=7, stride=7, padding=0,groups=6, bias=False)

        self.conv0= nn.Conv2d(6, self.cc, kernel_size=3, stride=2, padding=1,groups=6, bias=True)
        # self.bn0=nn.BatchNorm2d(self.cc//6)
        self.bn0=nn.BatchNorm2d(self.cc)
        # self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2, padding=1)
        self.conv1= nn.Conv2d(self.cc, self.cc, kernel_size=3, stride=2, padding=1,groups=6, bias=True)
        # self.bn0=nn.BatchNorm2d(self.cc//6)
        self.bn1=nn.BatchNorm2d(self.cc)
        
        self.csm= trans_conv( self.cc,  self.cc, kernelsize=3,padding=1, stride=1, size=(32,32 ))                       #通道之间关系

        
        
        self.relu=nn.ReLU()
        self.layer1= self._make_layer( BasicBlock,self.cc ,2 , stride=2)
        self.layer2= self._make_layer( BasicBlock,self.cc*2 ,2 , stride=2)
        self.layer3= self._make_layer( BasicBlock,self.cc*4 ,2 , stride=2)
        self.layer4= self._make_layer( BasicBlock,self.cc*8 ,2 , stride=2)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        # self.avgpool=nn.AdaptiveMaxPool2d(1)
        # self.lstm1=nn.LSTM( ,)

        self.fc_final=nn.Linear( self.cc*8,num_classes )

        # self.lstm1=nn.LSTM( self.cc*8,self.cc*8,2,batch_first=True,bidirectional=True )
        # self.lstm2=nn.LSTM( self.cc*8,self.cc*8,2,batch_first=True,bidirectional=True )
        # self.dropout=
        # self.lstm1=nn.LSTM( cc*8,cc*8,2,batch_first=True,bidirectional=True )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                # conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.Conv2d(self.inplanes, planes * block.expansion,kernel_size=1,stride= stride),
                # nn.BatchNorm2d(planes * block.expansion//6),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)



    def forward(self, input ):
                
        
        # for i in range(input.shape[1]):
            # x=input[:,i,:,:].reshape(input.shape[0],1,input.shape[2],input.shape[3])
            # .repeat(1,3,1,1)
            # with torch.no_grad():


        x = self.conv0(input)
        # a,b,c,d=x.shape
        # x=x.reshape(a,b//6 ,6  , c,d).reshape(a,b//6 ,6,-1 )
        # x = self.bn0(x).reshape(a,b//6,6,c,d  ).reshape( a,b,c,d)
        x = self.bn0(x)
        x = self.relu(x)
        

        x = self.conv1(x)
        x=self.gelu(x)
        
        x = self.bn1(x)
        
        x=self.csm(x)
        
        # x = self.relu(x)

        
        # x = self.conv1(input)
        # x = self.bn1(x)
        # x = self.relu(x)
        
        
        # x = self.maxpool(x)


        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x = self.avgpool(x)
        
        

        # x=x.reshape(x.shape[0],-1)
        # x=self.fc_final(x)


        # y=x.reshape(x.shape[0],6,-1)
        # x1=self.lstm1(x1)
        # print(x1.shape)
        # x1=self.dropout(x1)
        # x1=self.lstm1(x1)
        # print(x1.shape)
        # x1=self.dropout(x1)
        # self.fc_final=nn.Linear( self.cc*8,1 )

        #patch

        x=x.reshape(x.shape[0],6,self.dim,-1).permute(0,1,3,2).reshape( x.shape[0],-1, self.dim)                        #多头注意力机制
        # X.shape=(num_batch, num_qkv, num_heads, num_hiddens_single)#调整顺序后# X.shape=(num_batch, num_heads, num_qkv, num_hiddens_single)

        # y=x.reshape(x.shape[0],6,-1)
        x=self.norm0(x)
        # x1=self.fc0(x1)
        # x1=self.norm0_1(x1)
        b, n, _ = x.shape
        # cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = x1.shape[0])
        # x1 = torch.cat((cls_tokens, x1), dim=1)
        x += self.pos_embedding[:, :n ]                                                                                 #添加位置信息
        x = self.dropout(x)
        for attn, ff in self.layers:                                                                                    #transformer 全图各模块之间的关系
            x = attn(x) + x
            x = ff(x) + x
            
        x = x.mean(dim = 1)  
        x = self.to_latent(x)
        x= self.mlp_head(x)




        return x




class ResNet34(nn.Module):
    def __init__(self, model, num_classes=1000):
        super(ResNet34, self).__init__()
        self.backbone = model

        self.fc1 = nn.Linear(2048, 1024)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # x = whitening(x)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        
        x = x.view(x.size(0), -1)
        x = l2_norm(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = l2_norm(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

class ResNet50(nn.Module):
    def __init__(self, model, num_classes=1000):
        super(ResNet50, self).__init__()
        self.backbone = model

        self.fc1 = nn.Linear(8192, 2048)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(2048, num_classes)


    def forward(self, x):
        #x = whitening(x)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        
        x = x.view(x.size(0), -1)
        x = l2_norm(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = l2_norm(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class ResNet101(nn.Module):
    def __init__(self, model, num_classes=1000):
        super(ResNet101, self).__init__()
        self.backbone = model

        self.fc1 = nn.Linear(8192, 2048)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(2048, num_classes)

    def forward(self, x):
        #x = whitening(x)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        
        x = x.view(x.size(0), -1)
        x = l2_norm(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = l2_norm(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

class ResNet152(nn.Module):
    def __init__(self, model, num_classes=1000):
        super(ResNet152, self).__init__()
        self.backbone = model

        self.fc1 = nn.Linear(8192, 2048)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(2048, num_classes)


    def forward(self, x):
        #x = whitening(x)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        
        x = x.view(x.size(0), -1)
        x = l2_norm(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = l2_norm(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

if __name__ == '__main__':
    backbone = models.resnet101(pretrained=True)
    models = ResNet101(backbone, 21)
    data = torch.randn(1, 3, 256, 256)
    x = models(data)
    #print(x)
    print(x.size())
