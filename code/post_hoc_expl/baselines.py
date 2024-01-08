from torch.nn import Module

def get_baseline_list():
    return list(get_baseline_dict().keys())
    
def get_baseline_dict():
    return {"cam":{"const":CAM,"requ_targ":True}}

class CAM(Module):

    def __init__(self,model) -> None:
        super().__init__()
        self.model = model

    def forward(self,x,target):
        retDict = self.model(x)
        feats = retDict["feat"]
        
        weights = self.model.secondModel.linLay.weight.data[target]
        weights /= weights.sum(dim=1,keepdim=True)
        weights = weights.unsqueeze(-1).unsqueeze(-1)
        cam = (feats*weights).sum(dim=1,keepdim=True)
        return cam