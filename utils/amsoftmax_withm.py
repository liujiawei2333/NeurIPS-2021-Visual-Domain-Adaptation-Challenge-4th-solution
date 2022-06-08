import torch
import torch.nn as nn

class AMSoftmax(nn.Module):

    def __init__(self,
                 in_feats=1000,
                 n_classes=1000,
                 m=0.25,
                 s=30):
        super(AMSoftmax, self).__init__()
        self.m = m
        self.s = s
        self.in_feats = in_feats
        self.W = torch.nn.Parameter(torch.randn(in_feats, n_classes), requires_grad=True)
        # self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.W, gain=1)

    def forward(self, x, lb,mode):
        assert x.size()[0] == lb.size()[0]
        assert x.size()[1] == self.in_feats
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-9)
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-9)
        w_norm = torch.div(self.W, w_norm)
        costh = torch.mm(x_norm, w_norm)

        if mode == "eval":
            return costh * self.s
        elif mode == "train":
            delt_costh = torch.zeros_like(costh).scatter_(1, lb.unsqueeze(1), self.m)
            costh_m = costh - delt_costh
            costh_m_s = self.s * costh_m
            # loss = self.ce(costh_m_s, lb)
            return costh_m_s