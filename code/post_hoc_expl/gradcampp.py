#!/usr/bin/env python3
from typing import Any, Callable, List, Tuple, Union

import torch
import torch.nn.functional as F
from captum._utils.common import (
    _format_additional_forward_args,
    _format_output,
    _format_tensor_into_tuples,
)
from captum._utils.gradient import compute_layer_gradients_and_eval
from captum._utils.typing import TargetType
from captum.attr._utils.attribution import GradientAttribution, LayerAttribution
from captum.attr import LayerGradCam
from captum.log import log_usage
from torch import Tensor
from torch.nn import Module


class LayerGradCampp(LayerGradCam):

    def __init__(
        self,
        forward_func: Callable,
        layer: Module,
        device_ids: Union[None, List[int]] = None,
    ) -> None:

        LayerAttribution.__init__(self, forward_func, layer, device_ids)
        GradientAttribution.__init__(self, forward_func)

    @log_usage()
    def attribute(
        self,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        target: TargetType = None,
        additional_forward_args: Any = None,
        attribute_to_layer_input: bool = False,
        relu_attributions: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, ...]]:

        inputs = _format_tensor_into_tuples(inputs)
        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )
        # Returns gradient of output with respect to
        # hidden layer and hidden layer evaluated at each input.
        layer_gradients, layer_evals = compute_layer_gradients_and_eval(
            self.forward_func,
            self.layer,
            inputs,
            target,
            additional_forward_args,
            device_ids=self.device_ids,
            attribute_to_layer_input=attribute_to_layer_input,
        )

        squared_layer_gradients = tuple(
            layer_grad ** 2 for layer_grad in layer_gradients
        )
        cubed_layer_gradients = tuple(layer_grad ** 3 for layer_grad in layer_gradients)
        summed_acts = tuple(
            torch.sum(
                layer_eval,
                dim=tuple(x for x in range(2, len(layer_eval.shape))),
                keepdim=True,
            )
            for layer_eval in layer_evals
        )

        alphas = tuple(
            squared_layer_gradient
            / ((2 * squared_layer_gradient) + (cubed_layer_gradient * summed_act))
            for squared_layer_gradient, cubed_layer_gradient, summed_act in zip(
                squared_layer_gradients, cubed_layer_gradients, summed_acts
            )
        )

        # Replace NaNs with 0
        for alpha in alphas:
            alpha[alpha!=alpha] = 0

        summed_grads = tuple(
            torch.sum(
                alpha * F.relu(layer_grad),
                dim=tuple(x for x in range(2, len(layer_grad.shape))),
                keepdim=True,
            )
            for alpha, layer_grad in zip(alphas, layer_gradients)
        )

        scaled_acts = tuple(
            torch.sum(summed_grad * layer_eval, dim=1, keepdim=True)
            for summed_grad, layer_eval in zip(summed_grads, layer_evals)
        )
        if relu_attributions:
            scaled_acts = tuple(F.relu(scaled_act) for scaled_act in scaled_acts)
        return _format_output(len(scaled_acts) > 1, scaled_acts)
