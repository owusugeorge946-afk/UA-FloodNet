import argparse,random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from ablation_model import UAFloodNet,training_loss
from data import NPZFloodDataset


def scores(model,loader,device):
    tp=fp=fn=tn=0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch={k:v.to(device) for k,v in batch.items()}
            pred=model(batch['sar_pre'],batch['sar_post'],batch['ms_pre'],batch['ms_post'])['logits'].sigmoid()>=.5
            gt=batch['mask']>=.5
            tp+=(pred&gt).sum().item(); fp+=(pred&~gt).sum().item()
            fn+=(~pred&gt).sum().item(); tn+=(~pred&~gt).sum().item()
    return dict(flood_iou=tp/max(tp+fp+fn,1),flood_f1=2*tp/max(2*tp+fp+fn,1),precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),accuracy=(tp+tn)/max(tp+tn+fp+fn,1))



def translate_zero(x, dx, dy):
    _, _, h, w = x.shape
    result = torch.zeros_like(x)

    y0, y1 = max(0, dy), min(h, h + dy)
    x0, x1 = max(0, dx), min(w, w + dx)

    if y1 > y0 and x1 > x0:
        result[:, :, y0:y1, x0:x1] = x[
            :,
            :,
            y0-dy:y1-dy,
            x0-dx:x1-dx
        ]
    return result


def augment_misregistration(batch, probability, max_shift):
    """Independent paired translations for each sample."""
    if probability <= 0 or max_shift == 0:
        return batch

    pre = batch["ms_pre"]
    post = batch["ms_post"]

    pre_aug = []
    post_aug = []

    for i in range(pre.shape[0]):
        if torch.rand(()).item() < probability:
            dx = int(torch.randint(
                -max_shift, max_shift + 1, ()
            ).item())
            dy = int(torch.randint(
                -max_shift, max_shift + 1, ()
            ).item())
        else:
            dx, dy = 0, 0

        pre_aug.append(
            translate_zero(pre[i:i+1], dx, dy)
        )
        post_aug.append(
            translate_zero(post[i:i+1], dx, dy)
        )

    result = dict(batch)
    result["ms_pre"] = torch.cat(pre_aug, dim=0)
    result["ms_post"] = torch.cat(post_aug, dim=0)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--train',required=True);p.add_argument('--val',required=True)
    p.add_argument('--epochs',type=int,default=30);p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--lr',type=float,default=1e-3);p.add_argument('--width',type=int,default=24)
    p.add_argument('--variant',choices=['base','alignment','reliability','full'],required=True)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--out',default='runs/best.pt')
    p.add_argument('--contrast-weight',type=float,default=.1);p.add_argument('--recon-weight',type=float,default=.1)
    p.add_argument('--flow-weight',type=float,default=.01)
    p.add_argument('--shift-prob',type=float,default=0.0)
    p.add_argument('--max-shift',type=int,default=4)
    a=p.parse_args()
    if not 0 <= a.shift_prob <= 1:
        raise ValueError('shift-prob must be between 0 and 1')
    if a.max_shift < 0:
        raise ValueError('max-shift must be nonnegative')
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tr=DataLoader(NPZFloodDataset(a.train),batch_size=a.batch_size,shuffle=True,num_workers=0)
    va=DataLoader(NPZFloodDataset(a.val),batch_size=a.batch_size,shuffle=False,num_workers=0)
    use_alignment=a.variant in ('alignment','full')
    use_reliability=a.variant in ('reliability','full')
    model=UAFloodNet(width=a.width,use_alignment=use_alignment,use_reliability=use_reliability).to(device);opt=torch.optim.AdamW(model.parameters(),lr=a.lr)
    best=-1
    for epoch in range(1,a.epochs+1):
        model.train();total=0
        for batch in tr:
            batch={k:v.to(device) for k,v in batch.items()}
            batch=augment_misregistration(
                batch, a.shift_prob, a.max_shift
            )
            opt.zero_grad(set_to_none=True)
            out=model(batch['sar_pre'],batch['sar_post'],batch['ms_pre'],batch['ms_post'])
            loss,logs=training_loss(model,out,batch,a.contrast_weight,a.recon_weight,a.flow_weight)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            opt.step();total+=logs['loss']
        stats=scores(model,va,device)
        print(f'epoch={epoch} train_loss={total/len(tr):.4f} '+' '.join(f'{k}={v:.4f}' for k,v in stats.items()),flush=True)
        if stats['flood_iou']>best:
            best=stats['flood_iou'];Path(a.out).parent.mkdir(parents=True,exist_ok=True)
            torch.save({'model':model.state_dict(),'width':a.width,'epoch':epoch,'metrics':stats,'variant':a.variant,'use_alignment':use_alignment,'use_reliability':use_reliability,'seed':a.seed,'shift_prob':a.shift_prob,'max_shift':a.max_shift},a.out)
    print(f'Best validation flood IoU {best:.4f}, checkpoint {a.out}')
if __name__=='__main__':main()
