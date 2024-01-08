import torch 

def apply_softmax(tensor,inds):
    tensor = torch.softmax(tensor,dim=-1)
    inds = inds.unsqueeze(-1).unsqueeze(-1)
    inds = inds.expand(-1,tensor.shape[1],-1)
    tensor = tensor.gather(2,inds).squeeze(-1)
    return tensor

class RISE(torch.nn.Module):

    def __init__(self,model,nbMasks=8000,batchSize=1,res=14):
        super().__init__()

        print("RISE RES",res,"NB_MASKS",nbMasks,"BATCH_SIZE",batchSize)

        self.nbMasks = nbMasks
        self.batchSize = batchSize
        self.model = model.eval() 
        self.res = res

    def forward(self,x,target=None):

        x_cpu = x
        if target is None:
            inds = torch.argmax(self.model(x_cpu)["output"],dim=-1)
        else:
            inds = target

        totalMaskNb = 0
        masks = torch.zeros(self.batchSize,1,self.res,self.res).to(x.device)

        batchNb = 0

        allMasks,allOut = None,None
        out = None

        while totalMaskNb < self.nbMasks:
   
            masks = masks.bernoulli_()

            masks_ = torch.nn.functional.interpolate(masks,size=(x.size(-1)),mode="nearest")
            masks_ = masks_.unsqueeze(0)
            x_ = x.unsqueeze(1)

            x_mask = x_*masks_
            x_mask = x_mask.reshape(x.shape[0]*masks.shape[0],x.shape[1],x.shape[2],x.shape[3])

            resDic = self.model(x_mask)

            out = resDic["output"]
            out = out.reshape(x.shape[0],masks.shape[0],out.shape[1])
            out = apply_softmax(out,inds)
            
            allMasks = masks.cpu() if allMasks is None else torch.cat((allMasks,masks.cpu()),dim=0)
            allOut =  out.cpu() if allOut is None else torch.cat((allOut,out.cpu()),dim=1)

            totalMaskNb += self.batchSize

            batchNb += 1

        allOut = allOut.unsqueeze(-1).unsqueeze(-1)
        allMasks = allMasks.permute(1,0,2,3)
        salMap = (allMasks*allOut).sum(dim=1,keepdim=True)/(allMasks.mean(dim=1,keepdim=True)*totalMaskNb)

        return salMap
