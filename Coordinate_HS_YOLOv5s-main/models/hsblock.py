import torch
import torch.nn as nn

#stride=1的hsblock
class hs_block1(nn.Module):
    def __init__(self,  inp_dim, s=4):
        super(hs_block1, self).__init__()
        self.s = s
        self.inp_dim = inp_dim
        gaplist = []
        gap0 = inp_dim // s
        gaplist.append(gap0)
        gap = gap0
        for _ in range(2, s):
            gap = gap0 + gap // 2
            gaplist.append(gap)
        self.gaplist = gaplist

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(gaplist[i], gaplist[i], (3, 3), padding=(1, 1), bias=False),
                nn.BatchNorm2d(gaplist[i]),
                nn.ReLU(inplace=True)
            ) for i in range(s - 1)])


    def forward(self, x):
        outputs = []
        gap = self.inp_dim // self.s
        #将input分成等分
        a = torch.split(x,split_size_or_sections=gap ,dim=1)
        #第一组直接送到输出
        outputs.append(a[0])
        #第二组直接卷积然后平均分
        b = self.convs[0](a[1])
        b = torch.split(b, split_size_or_sections=self.gaplist[0] // 2, dim=1)
        outputs.append(b[0])
        #第三组开始就是拼接-卷积-均分的重复步骤了
        for i in range(2,self.s-1):
            b = torch.cat((a[i],b[1]),dim=1)
            b = self.convs[i-1](b)
            b = torch.split(b,split_size_or_sections=self.gaplist[i-1]//2 ,dim=1)
            outputs.append(b[0])
        #最后一组只有拼接-卷积，而不再均分了
        b = torch.cat((a[-1], b[1]), dim=1)
        b = self.convs[-1](b)
        outputs.append(b)
        #最后把上面的几个部分全部拼接在一起即可
        out = torch.cat(outputs, dim=1)


        return out

#stride=2的hsblock
class hs_block2(nn.Module):
    def __init__(self,  inp_dim, s=4):
        super(hs_block2, self).__init__()
        self.s = s
        self.inp_dim = inp_dim
        gaplist = []
        gap0 = inp_dim // s
        gaplist.append(gap0)
        gap = gap0
        for _ in range(2, s):
            gap = gap0 + gap // 2
            gaplist.append(gap)
        self.gaplist = gaplist

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(gaplist[0]//2, gaplist[0]//2, (3, 3), padding=(1, 1), stride=2, bias=False),
                nn.BatchNorm2d(gaplist[0]//2),
                nn.ReLU(inplace=True)
            ) for i in range(s - 1)])

        self.pool = nn.AvgPool2d(kernel_size=3, stride=(2, 2), padding=(1, 1))


    def forward(self, x):
        outputs = []
        gap = self.inp_dim // self.s
        #将input分成等分
        a = torch.split(x,split_size_or_sections=gap ,dim=1)
        #第一组直接送到输出
        outputs.append(self.pool(a[0]))
        #第二组开始就是均分-卷积的重复步骤了
        for i in range(1,self.s):
            b = torch.split(a[i], split_size_or_sections=self.gaplist[0]//2, dim=1)
            outputs.append(self.convs[i-1](b[0]))
            outputs.append(self.pool(b[1]))

        #最后把上面的几个部分全部拼接在一起即可
        out = torch.cat(outputs, dim=1)

        return out


class get_hsblock(nn.Module):
    def __init__(self, inp_dim, s=4, stride=1):
        super(get_hsblock, self).__init__()
        self.stride = stride
        if self.stride == 1:
            self.hsblock = hs_block1(inp_dim, s)
        else:
            self.hsblock = hs_block2(inp_dim, s)

    def forward(self, x):
        #aa = self.hsblock(x)

        return self.hsblock(x)
#
#
# class get_hsblock(nn.Module):
#     def __init__(self, inp_dim, s=4):
#         super(get_hsblock, self).__init__()
#
#         self.hsblock = hs_block1(inp_dim, s)
#
#
#     def forward(self, x):
#         #aa = self.hsblock(x)
#
#         return self.hsblock(x)