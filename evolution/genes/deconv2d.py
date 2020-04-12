import torch.nn as nn
from .conv2d import Conv2d
from ..layers.convupsample import ConvUpsample
import numpy as np
from ..config import config
import math
from util import tools
from evolution.layers.pixelwise_norm import PixelwiseNorm


class Deconv2d(Conv2d):
    """Represents a convolution layer."""

    def __init__(self, out_channels=None, kernel_size=config.layer.conv2d.kernel_size, stride=2, activation_type="random", activation_params={}, normalize=True):
        super().__init__(out_channels, kernel_size, stride, activation_type=activation_type, activation_params=activation_params, normalize=normalize)
        self.output_padding = 1
        self._current_input_shape = None

    def setup(self):
        calc_out_channels = self.out_channels is None
        super().setup()
        if calc_out_channels:
            if config.layer.conv2d.random_out_channels:
                self.out_channels = 2 ** np.random.randint(config.layer.conv2d.min_channels_power, config.layer.conv2d.max_channels_power)
            else:
                self.out_channels = max(1, self.in_channels//2)
        if config.layer.conv2d.force_double:
            self.out_channels = min(self.out_channels, self.in_channels//2)

        if self.is_last_layer():
            self.out_channels = self.final_output_shape[0]
        if self.out_channels > self.in_channels:
            self.stride = 1
            self.output_padding = 0
        else:
            self.stride = 2
            self.output_padding = 1

    def changed(self):
        return super().changed() or self._current_input_shape != self.input_shape

    def _create_normalization(self):
        if config.gan.pixelwise_normalization:
            return PixelwiseNorm()
        else:
            return super()._create_normalization()

    def _create_phenotype(self, input_shape):
        # adjust output size
        output_size = None
        padding = self.kernel_size // 2
        if not isinstance(self.final_output_shape, int) and self.stride > 1:
            in_dimension = np.array(input_shape[2:])
            out_dimension = np.array(self.final_output_shape[1:])
            div = np.round(out_dimension/(in_dimension*2)).astype(np.int32)
            div[div == 0] = 1
            output_size = out_dimension//div
            if not self.next_layer:  # set the output size for the final layer
                output_size = out_dimension

            # output formula for transpose convolution: o = (i-1)*s - 2*p + k + op
            self.stride = int(math.ceil((output_size[0] - self.kernel_size)/max(1, in_dimension[0] - 1)))
            padding = int(math.ceil(((in_dimension[0] - 1) * self.stride - output_size[0] + self.kernel_size)/2))
            self.output_padding = int(output_size[0] - (in_dimension[0] - 1) * self.stride + 2*padding - self.kernel_size)
            print("output_size", output_size, div, out_dimension, in_dimension, self.kernel_size)
            print("stride", self.stride, padding, self.output_padding)
            if output_size is not None:
                output_size = [x.item() for x in output_size]
        self._current_input_shape = self.input_shape
        layer = ConvUpsample(
            self.in_channels, self.out_channels, self.kernel_size, self.stride, output_size,
            output_padding=self.output_padding, padding=padding, bias=not config.gan.batch_normalization)

        if self.module is None or not config.layer.resize_weights:
            if self.has_wscale():
                nn.init.normal_(layer.module.weight)
                if layer.module.bias is not None:
                    nn.init.zeros_(layer.module.bias)
            else:
                nn.init.xavier_uniform_(layer.module.weight)
        else:
            # resize and copy weights
            print("RESIZE DECONV")
            if layer.module.bias is not None and self.module.bias.size() != layer.module.bias.size():
                layer.module.bias = nn.Parameter(tools.resize_1d(self.module.bias, layer.module.bias.size()[0]))
            print(self.module.module.weight.size(), layer.module.weight.size())
            try:
                w = tools.resize_activations(self.module.module.weight, layer.module.weight.size())
                print(w.size())
                layer.module.weight = nn.Parameter(w)
                self.adjusted = True
            except Exception as e:
                print("error resizing weights")
                print(e)

        return layer

    def first_deconv(self):
        return not self.previous_layer or not isinstance(self.previous_layer, Deconv2d)

    def is_upsample(self):
        return self.out_channels < self.in_channels
