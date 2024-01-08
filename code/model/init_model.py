import torch

def getOptim(optimStr, lr,momentum,weightDecay,net):

    if optimStr != "AMSGrad":
        optimConst = getattr(torch.optim, optimStr)
        if optimStr == "SGD":
            kwargs = {'lr':lr,'momentum': momentum,"weight_decay":weightDecay}
        elif optimStr == "Adam":
            kwargs = {'lr':lr,"weight_decay":weightDecay}
        elif optimStr == "AdamW":
            kwargs = {'lr':lr,"weight_decay":weightDecay}
        else:
            raise ValueError("Unknown optimisation algorithm : {}".format(optimStr))
    else:
        optimConst = torch.optim.Adam
        kwargs = {'lr':lr,'amsgrad': True,"weight_decay":weightDecay}

    optim = optimConst(net.parameters(), **kwargs)

    return optim

def preprocessAndLoadParams(init_path,cuda,net,verbose=True):
    if verbose:
        print("Init from",init_path)

    params = torch.load(init_path, map_location="cpu" if not cuda else None)
    params = addOrRemoveModule(params,net)
    res = net.load_state_dict(params, False)
    # Depending on the pytorch version the load_state_dict() method can return the list of missing and unexpected parameters keys or nothing
    if not res is None:
        missingKeys, unexpectedKeys = res
        if len(missingKeys) > 0:
            print("missing keys")
            for key in missingKeys:
                print(key)
        if len(unexpectedKeys) > 0:
            print("unexpected keys")
            for key in unexpectedKeys:
                print(key)

    return net

def addOrRemoveModule(params,net):
    # Checking if the key of the model start with "module."
    startsWithModule = (list(net.state_dict().keys())[0].find("module.") == 0)

    if startsWithModule:
        paramsFormated = {}
        for key in params.keys():
            keyFormat = "module." + key if key.find("module") == -1 else key
            paramsFormated[keyFormat] = params[key]
        params = paramsFormated
    else:
        paramsFormated = {}
        for key in params.keys():
            keyFormat = key.split('.')
            if keyFormat[0] == 'module':
                keyFormat = '.'.join(keyFormat[1:])
            else:
                keyFormat = '.'.join(keyFormat)
            paramsFormated[keyFormat] = params[key]
        params = paramsFormated
    return params