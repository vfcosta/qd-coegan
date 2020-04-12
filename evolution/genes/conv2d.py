import torch.nn as nn
from .layer2d import Layer2D
import numpy as np
from ..config import config
from util import tools


class Conv2d(Layer2D):
    """Represents a convolution layer."""

    def __init__(self, out_channels=None, kernel_size=config.layer.conv2d.kernel_size, stride=2, activation_type="random", padding=1, activation_params={}, normalize=True, use_dropout=config.gan.dropout):
        super().__init__(activation_type=activation_type, activation_params=activation_params, normalize=normalize, use_dropout=use_dropout)
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = None
        self.out_channels = out_channels
        self.padding = padding

    def setup(self):
        super().setup()
        self.in_channels = self.input_shape[1] if self.input_shape and len(self.input_shape) > 2 else 1
        # limit the kernel size to the input shape
        if self.input_shape is not None:
            self.kernel_size = min(self.kernel_size, self.input_shape[2])
        if self.out_channels is None:
            if self.first_conv() or config.layer.conv2d.random_out_channels:
                self.out_channels = 2 ** np.random.randint(config.layer.conv2d.min_channels_power, config.layer.conv2d.max_channels_power+1)
            else:
                self.out_channels = min(2 * self.in_channels, 2**(config.layer.conv2d.max_channels_power+1))
        if config.layer.conv2d.force_double:
            self.out_channels = max(self.out_channels, self.in_channels*2)

    def changed(self):
        module_kernel_size = self.module.kernel_size
        if not isinstance(self.module.kernel_size, int):
            module_kernel_size = module_kernel_size[0]
        return self.module.out_channels != self.out_channels or self.module.in_channels != self.in_channels or \
               self.kernel_size != module_kernel_size

    def _create_phenotype(self, input_shape):
        module = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, padding=self.padding,
                           bias=not config.gan.batch_normalization)

        if self.module is None or not config.layer.resize_weights:
            if self.has_wscale():
                nn.init.normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            else:
                nn.init.xavier_uniform_(module.weight)
        elif self.kernel_size == self.module.kernel_size[0]:
            # resize and copy weights only when the kernel size was not changed
            print("RESIZE")
            if module.bias is not None and self.module.bias.size() != module.bias.size():
                module.bias = nn.Parameter(tools.resize_1d(self.module.bias, module.bias.size()[0]))
                print("bias", module.bias.size(), self.module.bias.size())
            try:
                w = tools.resize_activations(self.module.weight, module.weight.size())
                print(self.module.weight.size(), module.weight.size(), w.size())
                module.weight = nn.Parameter(w)
            except Exception as e:
                print("error resizing weights")
                print(e)

        return module

    def apply_mutation(self):
        self.out_channels = None
        self.setup()

    def first_conv(self):
        return not self.previous_layer or not isinstance(self.previous_layer, Conv2d)

    def __repr__(self):
        return self.__class__.__name__ + f"(in_channels={self.in_channels}, out_channels={self.out_channels}, " \
                                         f"kernel_size={self.kernel_size}, stride={self.stride}, " \
                                         f"activation_type={self.activation_type})"
