"""
Created on Wed Apr 29 16:11:20 2020
@author: Haofan Wang - github.com/haofanwang
"""
import sys

import torch
import torch.nn.functional as F

EPSILON = sys.float_info.epsilon

def min_max_normalisation(unorm_map):
    map_min = unorm_map.amin(dim=(1,2,3),keepdim=True)
    map_max = unorm_map.amax(dim=(1,2,3),keepdim=True)
    norm_map = (unorm_map - map_min)/(map_max - map_min+EPSILON)
    return norm_map 

class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        for module_pos, module in self.model.features._modules.items():
            x = module(x)  # Forward
            if int(module_pos) == self.target_layer:
                conv_output = x  # Save the convolution output on that layer
        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        x = x.view(x.size(0), -1)  # Flatten
        # Forward pass on the classifier
        x = self.model.classifier(x)
        return conv_output, x

class ScoreCam():
    """
        Produces class activation map
    """
    def __init__(self, model):
        self.model = model
        self.model.eval()
        # Define extractor

        target_layer = model.firstModel.featMod.layer4
        model.features = model.firstModel

        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, target=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        retDict = self.model(input_image)

        conv_output, model_output = retDict["feat"],retDict["output"]

        if target is None:
            target = torch.argmax(model_output,dim=-1)
        
        target = target.long()

        # Create empty numpy array for saliency_map
        saliency_map = torch.ones((conv_output.shape[0],1,conv_output.shape[2],conv_output.shape[3])).to(input_image.device)
        # Multiply each weight with its conv output and then, sum

        batch_inds = torch.arange(conv_output.shape[0])

        for i in range(conv_output.shape[1]):

            # Unsqueeze to 4D

            feature_map = conv_output[:,i:i+1, :, :]
            # Upsampling to input size
            feature_map = F.interpolate(feature_map, size=input_image.shape[2:], mode='nearest')
            if feature_map.max() == feature_map.min():
                continue

            # Scale between 0-1
            norm_feature_map = min_max_normalisation(feature_map)
            # Get the target score

            w = F.softmax(self.model(input_image*norm_feature_map)["output"].detach(),dim=1)[batch_inds,target]

            #inp = None
            sum_elem = w.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * conv_output[:,i:i+1, :, :]

            saliency_map += sum_elem

        saliency_map = torch.relu(saliency_map)
        saliency_map = min_max_normalisation(saliency_map)
        return saliency_map

if __name__ == '__main__':
    # Get params
    target_example = 0 
    (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
        get_example_params(target_example)
    # Score cam
    score_cam = ScoreCam(pretrained_model, target_layer=11)
    # Generate cam mask
    cam = score_cam.generate_cam(prep_img, target_class)
    # Save mask
    save_class_activation_images(original_image, cam, file_name_to_export)
    print('Score cam completed')
