"""Research prototype: uncertainty-aware, alignment-aware SAR/MS flood change detection.
Independent implementation inspired by DR-SGFN (Huang et al., IEEE JSTARS 2026).
Not an official reproduction; differs from the published architecture.
"""
import torch
from torch import nn
from torch.nn import functional as F


def conv_block(in_ch, out_ch):
    return nn.Sequential(nn.Conv2d(in_ch,out_ch,3,padding=1,bias=False),nn.BatchNorm2d(out_ch),nn.ReLU(inplace=True),
                         nn.Conv2d(out_ch,out_ch,3,padding=1,bias=False),nn.BatchNorm2d(out_ch),nn.ReLU(inplace=True))

class Encoder(nn.Module):
    def __init__(self,in_ch,width=24):
        super().__init__()
        self.b1=conv_block(in_ch,width)
        self.b2=conv_block(width,width*2)
        self.b3=conv_block(width*2,width*4)
    def forward(self,x):
        a=self.b1(x); b=self.b2(F.avg_pool2d(a,2)); c=self.b3(F.avg_pool2d(b,2))
        return [a,b,c]

class AlignFusion(nn.Module):
    """Estimate same-time optical->SAR displacement and per-sensor uncertainty.
    Learned log variances are proxies; assess calibration empirically.
    """
    def __init__(self,ch,max_shift=4.,use_alignment=True,use_reliability=True):
        super().__init__()
        self.max_shift=max_shift
        self.use_alignment=use_alignment
        self.use_reliability=use_reliability
        self.flow_net=nn.Sequential(conv_block(ch*2,ch),nn.Conv2d(ch,2,3,padding=1))
        self.logvar_net=nn.Sequential(conv_block(ch*2,ch),nn.Conv2d(ch,2,1))
        self.mix=conv_block(ch*2,ch)
    def forward(self,sar,ms):
        n,_,h,w=sar.shape
        flow=(self.max_shift*torch.tanh(self.flow_net(torch.cat([sar,ms],dim=1)))
              if self.use_alignment else sar.new_zeros((n,2,h,w)))
        yy,xx=torch.meshgrid(torch.arange(h,device=sar.device,dtype=sar.dtype),torch.arange(w,device=sar.device,dtype=sar.dtype),indexing='ij')
        x=xx[None]+flow[:,0]; y=yy[None]+flow[:,1]
        gx=2*x/max(w-1,1)-1; gy=2*y/max(h-1,1)-1
        grid=torch.stack([gx,gy],dim=-1)
        aligned=(F.grid_sample(ms,grid,align_corners=True,padding_mode='border')
                 if self.use_alignment else ms)
        # learn heteroscedastic sensor scores independently (normalization forces competition)
        if self.use_reliability:
            logvars=self.logvar_net(torch.cat([sar,aligned],dim=1)).clamp(-6,6)
            weights=torch.softmax(-logvars,dim=1)
        else:
            logvars=sar.new_zeros((n,2,h,w))
            weights=sar.new_full((n,2,h,w),0.5)
        fused=self.mix(torch.cat([sar*weights[:,0:1],aligned*weights[:,1:2]],dim=1))
        return fused,flow,weights,aligned,logvars

class UAFloodNet(nn.Module):
    def __init__(self,sar_channels=1,ms_channels=3,width=24,max_shift=4.,use_alignment=True,use_reliability=True):
        super().__init__()
        # Two modality encoders, each reused across the two times (four inputs)
        self.sar_encoder=Encoder(sar_channels,width)
        self.ms_encoder=Encoder(ms_channels,width)
        self.fusers=nn.ModuleList([AlignFusion(width*(2**i),max_shift=max_shift/(2**i),
                                             use_alignment=use_alignment,use_reliability=use_reliability) for i in range(3)])
        self.temporal=conv_block(width*4*3,width*4)
        self.decode2=conv_block(width*4+width*2*3,width*2)
        self.decode1=conv_block(width*2+width*3,width)
        self.head=nn.Conv2d(width,1,1)
        # Auxiliary representation regularization used only by loss function
        self.project_sar=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(width*4,64))
        self.project_ms=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(width*4,64))
        self.reconstruct_sar=nn.Conv2d(width*4,sar_channels,1)
        self.reconstruct_ms=nn.Conv2d(width*4,ms_channels,1)
    def forward(self,sar_pre,sar_post,ms_pre,ms_post):
        sizes={tuple(t.shape[-2:]) for t in (sar_pre,sar_post,ms_pre,ms_post)}
        if len(sizes)!=1: raise ValueError('All inputs must have identical H,W; do initial georeferencing first')
        if min(sar_pre.shape[-2:])<8: raise ValueError('Images must be at least 8 x 8')
        s0=self.sar_encoder(sar_pre); s1=self.sar_encoder(sar_post)
        m0=self.ms_encoder(ms_pre); m1=self.ms_encoder(ms_post)
        fp=[];fq=[];flows=[]; weights=[]; logvars=[]
        for level,fuser in enumerate(self.fusers):
            p,flow_p,w_p,_,lv_p=fuser(s0[level],m0[level])
            q,flow_q,w_q,_,lv_q=fuser(s1[level],m1[level])
            fp.append(p);fq.append(q)
            flows.append((flow_p,flow_q));weights.append((w_p,w_q));logvars.append((lv_p,lv_q))
        # Change-preserving temporal differences; no cross-time warping
        d=[torch.abs(a-b) for a,b in zip(fp,fq)]
        high=self.temporal(torch.cat([fp[2],fq[2],d[2]],1))
        z2=F.interpolate(high,size=d[1].shape[-2:],mode='bilinear',align_corners=False)
        z2=self.decode2(torch.cat([z2,fp[1],fq[1],d[1]],1))
        z1=F.interpolate(z2,size=d[0].shape[-2:],mode='bilinear',align_corners=False)
        z1=self.decode1(torch.cat([z1,fp[0],fq[0],d[0]],1))
        logits=self.head(z1)
        return {'logits':logits,'flows':flows,'weights':weights,'logvars':logvars,
                'deep_sar':(s0[-1],s1[-1]),'deep_ms':(m0[-1],m1[-1])}


def info_nce(s,m,temperature=.1):
    # Within-batch cross-modal positive pairs. Batch 1: no valid negatives, skip.
    if s.shape[0]<2: return s.sum()*0.
    s=F.normalize(s,dim=-1);m=F.normalize(m,dim=-1)
    logits=s@m.T/temperature; labels=torch.arange(s.size(0),device=s.device)
    return .5*(F.cross_entropy(logits,labels)+F.cross_entropy(logits.T,labels))


def training_loss(model,output,batch,contrast_weight=.1,recon_weight=.1,flow_weight=.01):
    truth=batch['mask'].float()
    logits=output['logits']
    bce=F.binary_cross_entropy_with_logits(logits,truth)
    prob=logits.sigmoid(); dice=1-(2*(prob*truth).sum((1,2,3))+1)/((prob+truth).sum((1,2,3))+1)
    supervised=bce+dice.mean()
    con=logits.sum()*0.; rec=logits.sum()*0.
    for time,sar_key,ms_key in ((0,'sar_pre','ms_pre'),(1,'sar_post','ms_post')):
        s=output['deep_sar'][time];m=output['deep_ms'][time]
        con=con+info_nce(model.project_sar(s),model.project_ms(m))*.5
        # Cross-reconstruct the opposite modality at deepest feature scale
        ms_target=F.interpolate(batch[ms_key],size=s.shape[-2:],mode='area')
        sar_target=F.interpolate(batch[sar_key],size=s.shape[-2:],mode='area')
        rec=rec+.25*(F.mse_loss(model.reconstruct_ms(s),ms_target)+F.mse_loss(model.reconstruct_sar(m),sar_target))
    smooth=logits.sum()*0.
    for flow_pair in output['flows']:
        for f in flow_pair:
            smooth=smooth+(f[:,:,:,1:]-f[:,:,:,:-1]).abs().mean()+(f[:,:,1:,:]-f[:,:,:-1,:]).abs().mean()
    smooth=smooth/(len(output['flows'])*2)
    loss=supervised+contrast_weight*con+recon_weight*rec+flow_weight*smooth
    return loss,{'loss':float(loss.detach()),'seg':float(supervised.detach()),'contrast':float(con.detach()),'recon':float(rec.detach()),'flow_smooth':float(smooth.detach())}
