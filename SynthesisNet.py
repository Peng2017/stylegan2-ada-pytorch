import sys
from turtle import forward
sys.path.append(r"/data/stylegan_xl/stylegan_xl")
import numpy as np
import torch
from torch_utils import misc
from torch_utils import persistence
from torch_utils.ops import conv2d_resample
from torch_utils.ops import upfirdn2d
from torch_utils.ops import bias_act
from torch_utils.ops import fma
import dnnlib
import pickle
from torch_utils import legacy
import peng_network
import PIL.Image

# torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = False

@misc.profiled_function
def modulated_conv2d(
    x,                          # Input tensor of shape [batch_size, in_channels, in_height, in_width].
    weight,                     # Weight tensor of shape [out_channels, in_channels, kernel_height, kernel_width].
    styles,                     # Modulation coefficients of shape [batch_size, in_channels].
    noise           = None,     # Optional noise tensor to add to the output activations.
    up              = 1,        # Integer upsampling factor.
    down            = 1,        # Integer downsampling factor.
    padding         = 0,        # Padding with respect to the upsampled image.
    resample_filter = None,     # Low-pass filter to apply when resampling activations. Must be prepared beforehand by calling upfirdn2d.setup_filter().
    demodulate      = True,     # Apply weight demodulation?
    flip_weight     = True,     # False = convolution, True = correlation (matches torch.nn.functional.conv2d).
    fused_modconv   = True,     # Perform modulation, convolution, and demodulation as a single fused operation?
):
    batch_size = x.shape[0]
    out_channels, in_channels, kh, kw = weight.shape
    misc.assert_shape(weight, [out_channels, in_channels, kh, kw]) # [OIkk]
    misc.assert_shape(x, [batch_size, in_channels, None, None]) # [NIHW]
    misc.assert_shape(styles, [batch_size, in_channels]) # [NI]

    # Pre-normalize inputs to avoid FP16 overflow.
    if x.dtype == torch.float16 and demodulate:
        weight = weight * (1 / np.sqrt(in_channels * kh * kw) / weight.norm(float('inf'), dim=[1,2,3], keepdim=True)) # max_Ikk
        styles = styles / styles.norm(float('inf'), dim=1, keepdim=True) # max_I

    # Calculate per-sample weights and demodulation coefficients.
    w = None
    dcoefs = None
    if demodulate or fused_modconv:
        w = weight.unsqueeze(0) # [NOIkk]
        w = w * styles.reshape(batch_size, 1, -1, 1, 1) # [NOIkk]
    if demodulate:
        dcoefs = (w.square().sum(dim=[2,3,4]) + 1e-8).rsqrt() # [NO]
    if demodulate and fused_modconv:
        w = w * dcoefs.reshape(batch_size, -1, 1, 1, 1) # [NOIkk]

    # Execute by scaling the activations before and after the convolution.
    if not fused_modconv:
        x = x * styles.to(x.dtype).reshape(batch_size, -1, 1, 1)
        x = conv2d_resample.conv2d_resample(x=x, w=weight.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, flip_weight=flip_weight)
        if demodulate and noise is not None:
            x = fma.fma(x, dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1), noise.to(x.dtype))
        elif demodulate:
            x = x * dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1)
        elif noise is not None:
            x = x.add_(noise.to(x.dtype))
        return x

    # Execute as one fused op using grouped convolution.
    batch_size = int(batch_size)
    misc.assert_shape(x, [batch_size, in_channels, None, None])
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw)
    x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, groups=batch_size, flip_weight=flip_weight)
    x = x.reshape(batch_size, -1, *x.shape[2:])
    if noise is not None:
        x = x.add_(noise)
    return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class FullyConnectedLayer(torch.nn.Module): # activation='liner'仅对synthesisnetwork使用，对mapping和discriminator 不适用
    def __init__(self,in_features,out_features, bias_init,layername):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(out_features, 512))
        self.bias = torch.nn.Parameter(torch.full([out_features], 1.0)) 
        print(f'''
self.{layername}_weight = torch.nn.Parameter(torch.randn({out_features}, 512))
self.{layername}_bias = torch.nn.Parameter(torch.full([{out_features}], 1.0)) 
torch.addmm(self.{layername}_bias.to(ws.dtype).unsqueeze(0), ws[:,,:], (self.{layername}_weight.to(ws.dtype) * 0.044194173824159216).t())
        ''')
    def forward(self, x):
        return torch.addmm(self.bias.to(x.dtype).unsqueeze(0), x, (self.weight.to(x.dtype) * 0.044194173824159216).t())

@persistence.persistent_class
class SynthesisLayer(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        w_dim,                          # Intermediate latent (W) dimensionality.
        resolution,                     # Resolution of this layer.
        kernel_size     = 3,            # Convolution kernel size.
        up              = 1,            # Integer upsampling factor.
        use_noise       = True,         # Enable noise input?
        conv_clamp      = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        channels_last   = False,        # Use channels_last format for the weights?
        resample_filter =[1,3,3,1],
        layername=None
    ):
        super().__init__()
        self.resolution = resolution
        self.up = up
        self.use_noise = use_noise
        self.activation = 'lrelu'
        self.conv_clamp = 256
        self.register_buffer('resample_filter', upfirdn2d.setup_filter([1,3,3,1]))
        self.padding = kernel_size // 2
        self.act_gain = bias_act.activation_funcs['lrelu'].def_gain

        self.affine = FullyConnectedLayer(512, in_channels, bias_init=1,layername=layername+"_affine")
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size).to(memory_format=memory_format))
        if use_noise:
            self.register_buffer('noise_const', torch.randn(resolution, resolution))
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        #################################
#         assert use_noise
#         assert not channels_last
#         print(f'''
# self.{layername}_resolution = {resolution}
# self.{layername}_up = {up}
# self.{layername}_padding = {kernel_size}//2
# self.{layername}_affine = FullyConnectedLayer(512, {in_channels}, bias_init=1)
# self.{layername}_weight = torch.nn.Parameter(torch.randn({out_channels}, {in_channels}, {kernel_size}, {kernel_size}).to(memory_format=torch.contiguous_format))
# self.register_buffer('{layername}_noise_const', torch.randn({resolution}, {resolution}))
# self.{layername}_noise_strength = torch.nn.Parameter(torch.zeros([]))
# self.{layername}_bias = torch.nn.Parameter(torch.zeros([{out_channels}]))
#         ''')
    def forward(self, x, w, noise_mode='random', fused_modconv=True):
        in_resolution = self.resolution // self.up
        misc.assert_shape(x, [None, self.weight.shape[1], in_resolution, in_resolution])
        styles = self.affine(w)

        noise = None
        if self.use_noise and noise_mode == 'random':
            noise = torch.randn([x.shape[0], 1, self.resolution, self.resolution], device=x.device) * self.noise_strength
        if self.use_noise and noise_mode == 'const':
            noise = self.noise_const * self.noise_strength

        flip_weight = (self.up == 1) # slightly faster
        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, noise=noise, up=self.up,
            padding=self.padding, resample_filter=self.resample_filter, flip_weight=flip_weight, fused_modconv=fused_modconv)

        x = bias_act.bias_act(x, self.bias.to(x.dtype), act='lrelu', gain=self.act_gain, clamp=256)
        return x


@persistence.persistent_class
class ToRGBLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels=3, w_dim=512, kernel_size=1, conv_clamp=256, channels_last=False,layername=None):
        super().__init__()
        self.conv_clamp = 256
        self.affine = FullyConnectedLayer(512, in_channels, bias_init=1,layername=layername+"_affine")
        self.weight = torch.nn.Parameter(torch.randn([3, in_channels, 1, 1]).to(memory_format=torch.contiguous_format))
        self.bias = torch.nn.Parameter(torch.zeros([3]))
        self.weight_gain = 1 / np.sqrt(in_channels)

    def forward(self, x, w, fused_modconv=True):
        styles = self.affine(w) * self.weight_gain
        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, demodulate=False, fused_modconv=fused_modconv)
        x = bias_act.bias_act(x, self.bias.to(x.dtype), clamp=256)
        return x

@persistence.persistent_class
class SynthesisBlock(torch.nn.Module):
    def __init__(self,
        in_channels,                        # Number of input channels, 0 = first block.
        out_channels,                       # Number of output channels.
        w_dim,                              # Intermediate latent (W) dimensionality.
        resolution,                         # Resolution of this block.
        img_channels,                       # Number of output color channels.
        is_last,                            # Is this the last block?
        conv_clamp          = 256,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16            = False,        # Use FP16 for this block?
        layername=None
    ):
        super().__init__()
        self.in_channels = in_channels
        self.w_dim = 512
        self.resolution = resolution
        self.img_channels = 3
        self.is_last = resolution==1024
        self.architecture = 'skip'
        self.use_fp16 = resolution>=128
        self.channels_last = False
        self.register_buffer('resample_filter', upfirdn2d.setup_filter([1,3,3,1]))
        ###############################################
        # print(f'''
        # self.{layername}_in_channels = {in_channels}
        # self.{layername}_resolution = {resolution}
        # self.{layername}_is_last = {resolution==1024}
        # self.{layername}_use_fp16 = {resolution>=128}
        # ''')
        ######################################################
        if in_channels == 0:
            self.const = torch.nn.Parameter(torch.randn([out_channels, resolution, resolution]))
            self.conv1 = SynthesisLayer(out_channels, out_channels, w_dim=512, resolution=resolution,conv_clamp=256, channels_last=self.channels_last,layername=layername+"_conv1")
            self.torgb = ToRGBLayer(out_channels, img_channels, w_dim=512,conv_clamp=256, channels_last=self.channels_last,layername=layername+"_torgb")
            self.num_conv=1
            self.num_torgb=1
            # print(f'''
            # self.{layername}_const = torch.nn.Parameter(torch.randn({out_channels}, {resolution}, {resolution}))
            # self.{layername}_conv1 = SynthesisLayer({out_channels}, {out_channels}, w_dim=512, resolution={resolution},conv_clamp=256, channels_last={self.channels_last})
            # self.{layername}_torgb = ToRGBLayer({out_channels}, {img_channels}, w_dim=512,conv_clamp=256, channels_last={self.channels_last})
            # self.{layername}_num_conv=1
            # self.{layername}_num_torgb=1
            # ''')

        if in_channels != 0:
            self.conv0 = SynthesisLayer(in_channels, out_channels, w_dim=512, resolution=resolution, up=2,resample_filter=[1,3,3,1], conv_clamp=256, channels_last=self.channels_last,layername=layername+"_conv0")
            self.conv1 = SynthesisLayer(out_channels, out_channels, w_dim=512, resolution=resolution,conv_clamp=256, channels_last=self.channels_last,layername=layername+"_conv1")
            self.torgb = ToRGBLayer(out_channels, img_channels, w_dim=512,conv_clamp=256, channels_last=self.channels_last,layername=layername+"_torgb")
            self.num_conv=2
            self.num_torgb=1
            # print(f'''
            # self.{layername}_conv0 = SynthesisLayer({in_channels}, {out_channels}, w_dim=512, resolution={resolution}, up=2,resample_filter=[1,3,3,1], conv_clamp=256, channels_last={self.channels_last})
            # self.{layername}_conv1 = SynthesisLayer({out_channels}, {out_channels}, w_dim=512, resolution={resolution},conv_clamp=256, channels_last={self.channels_last})
            # self.{layername}_torgb = ToRGBLayer({out_channels}, {img_channels}, w_dim=512,conv_clamp=256, channels_last={self.channels_last})
            # self.{layername}_num_conv=2
            # self.{layername}_num_torgb=1
            # ''')

    def forward(self, x, img, ws, force_fp32=False, noise_mode='random'):
        w_iter = iter(ws.unbind(dim=1))
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        fused_modconv = (not self.training) and (dtype == torch.float32 or int(x.shape[0]) == 1)

        # Input.
        if self.in_channels == 0:
            x = self.const.to(dtype=dtype, memory_format=torch.contiguous_format)
            x = x.unsqueeze(0).repeat([ws.shape[0], 1, 1, 1])
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, noise_mode=noise_mode)
            y = self.torgb(x, next(w_iter), fused_modconv=fused_modconv)
            img = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)

        else:
            x = x.to(dtype=dtype, memory_format=torch.contiguous_format)
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, noise_mode=noise_mode)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, noise_mode=noise_mode)
            img = upfirdn2d.upsample2d(img, self.resample_filter)
            y = self.torgb(x, next(w_iter), fused_modconv=fused_modconv)
            y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            img = img.add_(y) 
        
        return x, img

@persistence.persistent_class
class SynthesisNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w_dim = 512
        self.img_resolution = 1024
        self.img_resolution_log2 = 10
        self.img_channels = 3
        self.block_resolutions = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
        channels_dict = {4: 512, 8: 512, 16: 512, 32: 512, 64: 512, 128: 256, 256: 128, 512: 64, 1024: 32}
        fp16_resolution = 128
        self.num_ws = 18
        self.b4=SynthesisBlock(in_channels=0, out_channels=512, w_dim=512, resolution=4,img_channels=3, is_last=False, use_fp16=False, conv_clamp=256,layername='b4') # num_conv=1 num_torgb=1
        self.b8=SynthesisBlock(in_channels=512, out_channels=512, w_dim=512, resolution=8,img_channels=3, is_last=False, use_fp16=False, conv_clamp=256,layername='b8')# num_conv=2 num_torgb=1
        self.b16=SynthesisBlock(in_channels=512, out_channels=512, w_dim=512, resolution=16,img_channels=3, is_last=False, use_fp16=False, conv_clamp=256,layername='b16')# num_conv=2 num_torgb=1
        self.b32=SynthesisBlock(in_channels=512, out_channels=512, w_dim=512, resolution=32,img_channels=3, is_last=False, use_fp16=False, conv_clamp=256,layername='b32')# num_conv=2 num_torgb=1
        self.b64=SynthesisBlock(in_channels=512, out_channels=512, w_dim=512, resolution=64,img_channels=3, is_last=False, use_fp16=False, conv_clamp=256,layername='b64')# num_conv=2 num_torgb=1
        self.b128=SynthesisBlock(in_channels=512, out_channels=256, w_dim=512, resolution=128,img_channels=3, is_last=False, use_fp16=True, conv_clamp=256,layername='b128')# num_conv=2 num_torgb=1
        self.b256=SynthesisBlock(in_channels=256, out_channels=128, w_dim=512, resolution=256,img_channels=3, is_last=False, use_fp16=True, conv_clamp=256,layername='b256')# num_conv=2 num_torgb=1
        self.b512=SynthesisBlock(in_channels=128, out_channels=64, w_dim=512, resolution=512,img_channels=3, is_last=False, use_fp16=True, conv_clamp=256,layername='b512')# num_conv=2 num_torgb=1
        self.b1024=SynthesisBlock(in_channels=64, out_channels=32, w_dim=512, resolution=1024,img_channels=3, is_last=True, use_fp16=True, conv_clamp=256,layername='b1024')# num_conv=2 num_torgb=1

    def forward(self, ws, force_fp32=False, noise_mode='random'):
        ws = ws.to(torch.float32)
        x, img = self.b4(None, None, ws.narrow(1, 0, 2), force_fp32=True, noise_mode=noise_mode)
        x, img = self.b8(x, img, ws.narrow(1, 1, 3), force_fp32=True, noise_mode=noise_mode)
        x, img = self.b16(x, img, ws.narrow(1, 3, 3), force_fp32=True, noise_mode=noise_mode)
        x, img = self.b32(x, img, ws.narrow(1, 5, 3), force_fp32=True, noise_mode=noise_mode)
        x, img = self.b64(x, img, ws.narrow(1, 7, 3), force_fp32=True, noise_mode=noise_mode)
        x, img = self.b128(x, img, ws.narrow(1, 9, 3), force_fp32=force_fp32, noise_mode=noise_mode)
        x, img = self.b256(x, img, ws.narrow(1, 11, 3), force_fp32=force_fp32, noise_mode=noise_mode)
        x, img = self.b512(x, img, ws.narrow(1, 13, 3), force_fp32=force_fp32, noise_mode=noise_mode)
        x, img = self.b1024(x, img, ws.narrow(1, 15, 3), force_fp32=force_fp32, noise_mode=noise_mode)
        return img

class SynthesisNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w_dim = 512
        self.img_channels = 3
        self.architecture = 'skip'
        self.channels_last = False
        # self.register_buffer('resample_filter', upfirdn2d.setup_filter([1,3,3,1]))
        self.register_buffer('resample_filter',torch.tensor([
                                [0.015625,0.046875,0.046875,0.015625],
                                [0.046875,0.140625, 0.140625, 0.046875],
                                [0.046875, 0.140625, 0.140625, 0.046875],
                                [0.015625, 0.046875, 0.046875, 0.015625]]).to(torch.float32))
        # b4
        self.b4_in_channels = 0
        self.b4_resolution = 4
        self.b4_is_last = False
        self.b4_use_fp16 = False
        self.b4_num_conv=1
        self.b4_num_torgb=1
        self.b4_const = torch.nn.Parameter(torch.randn(512, 4, 4)).to(dtype=torch.float32, memory_format=torch.contiguous_format)
        #self.b4_conv1 = SynthesisLayer(512, 512, w_dim=512, resolution=4,conv_clamp=256, channels_last=False)
        self.b4_conv1_resolution = 4
        self.b4_conv1_up = 1
        self.b4_conv1_padding = 3//2
        # self.b4_conv1_affine = FullyConnectedLayer(512, 512, bias_init=1)
        self.b4_conv1_affine_weight = torch.nn.Parameter(torch.randn(512, 512))
        self.b4_conv1_affine_bias = torch.nn.Parameter(torch.full([512], 1.0)) 

        self.b4_conv1_weight = torch.nn.Parameter(torch.randn(512, 512, 3, 3).to(memory_format=torch.contiguous_format))
        self.register_buffer('b4_conv1_noise_const', torch.randn(4, 4))
        self.b4_conv1_noise_strength = torch.nn.Parameter(torch.zeros([]))
        self.b4_conv1_bias = torch.nn.Parameter(torch.zeros([512]))
        # self.b4_torgb = ToRGBLayer(512, 3, w_dim=512,conv_clamp=256, channels_last=False)
        # self.b4_torgb_affine = FullyConnectedLayer(512, 512, bias_init=1)
        self.b4_torgb_affine_weight = torch.nn.Parameter(torch.randn(512, 512))
        self.b4_torgb_affine_bias = torch.nn.Parameter(torch.full([512], 1.0)) 

        self.b4_torgb_weight = torch.nn.Parameter(torch.randn(3, 512, 1, 1).to(memory_format=torch.contiguous_format))
        self.b4_torgb_bias = torch.nn.Parameter(torch.zeros([3]))
        self.b4_torgb_weight_gain = 1 / np.sqrt(512)
        
        # b8
        self.b8_in_channels = 512
        self.b8_resolution = 8
        self.b8_is_last = False
        self.b8_use_fp16 = False
        self.b8_num_conv=2
        self.b8_num_torgb=1
        # self.b8_conv0 = SynthesisLayer(512, 512, w_dim=512, resolution=8, up=2,resample_filter=[1,3,3,1], conv_clamp=256, channels_last=False)
        self.b8_conv0_resolution = 8
        self.b8_conv0_up = 2
        self.b8_conv0_padding = 3//2
        # self.b8_conv0_affine = FullyConnectedLayer(512, 512, bias_init=1)
        self.b8_conv0_affine_weight = torch.nn.Parameter(torch.randn(512, 512))
        self.b8_conv0_affine_bias = torch.nn.Parameter(torch.full([512], 1.0)) 

        self.b8_conv0_weight = torch.nn.Parameter(torch.randn(512, 512, 3, 3).to(memory_format=torch.contiguous_format))
        self.register_buffer('b8_conv0_noise_const', torch.randn(8, 8))
        self.b8_conv0_noise_strength = torch.nn.Parameter(torch.zeros([]))
        self.b8_conv0_bias = torch.nn.Parameter(torch.zeros([512]))
        # self.b8_conv1 = SynthesisLayer(512, 512, w_dim=512, resolution=8,conv_clamp=256, channels_last=False)
        self.b8_conv1_resolution = 8
        self.b8_conv1_up = 1
        self.b8_conv1_padding = 3//2
        # self.b8_conv1_affine = FullyConnectedLayer(512, 512, bias_init=1)
        self.b8_conv1_affine_weight = torch.nn.Parameter(torch.randn(512, 512))
        self.b8_conv1_affine_bias = torch.nn.Parameter(torch.full([512], 1.0))

        self.b8_conv1_weight = torch.nn.Parameter(torch.randn(512, 512, 3, 3).to(memory_format=torch.contiguous_format))
        self.register_buffer('b8_conv1_noise_const', torch.randn(8, 8))
        self.b8_conv1_noise_strength = torch.nn.Parameter(torch.zeros([]))
        self.b8_conv1_bias = torch.nn.Parameter(torch.zeros([512]))
        # self.b8_torgb = ToRGBLayer(512, 3, w_dim=512,conv_clamp=256, channels_last=False)
        # self.b8_torgb_affine = FullyConnectedLayer(512, 512, bias_init=1)
        self.b8_torgb_affine_weight = torch.nn.Parameter(torch.randn(512, 512))
        self.b8_torgb_affine_bias = torch.nn.Parameter(torch.full([512], 1.0)) 

        self.b8_torgb_weight = torch.nn.Parameter(torch.randn(3, 512, 1, 1).to(memory_format=torch.contiguous_format))
        self.b8_torgb_bias = torch.nn.Parameter(torch.zeros([3]))
        self.b8_torgb_weight_gain = 1 / np.sqrt(512)
        
        # b16
        self.b16_in_channels = 512
        self.b16_resolution = 16
        self.b16_is_last = False
        self.b16_use_fp16 = False
        # self.b16_conv0 = SynthesisLayer(512, 512, w_dim=512, resolution=16, up=2,resample_filter=[1,3,3,1], conv_clamp=256, channels_last=False)
        self.b16_conv0_resolution = 16
        self.b16_conv0_up = 2
        self.b16_conv0_padding = 3//2
        # self.b16_conv0_affine = FullyConnectedLayer(512, 512, bias_init=1)
        self.b16_conv0_affine_weight = torch.nn.Parameter(torch.randn(512, 512))
        self.b16_conv0_affine_bias = torch.nn.Parameter(torch.full([512], 1.0)) 

        self.b16_conv0_weight = torch.nn.Parameter(torch.randn(512, 512, 3, 3).to(memory_format=torch.contiguous_format))
        self.register_buffer('b16_conv0_noise_const', torch.randn(16, 16))
        self.b16_conv0_noise_strength = torch.nn.Parameter(torch.zeros([]))
        self.b16_conv0_bias = torch.nn.Parameter(torch.zeros([512]))
        # self.b16_conv1 = SynthesisLayer(512, 512, w_dim=512, resolution=16,conv_clamp=256, channels_last=False)
        self.b16_conv1_resolution = 16
        self.b16_conv1_up = 1
        self.b16_conv1_padding = 3//2
        # self.b16_conv1_affine = FullyConnectedLayer(512, 512, bias_init=1)
        self.b16_conv1_affine_weight = torch.nn.Parameter(torch.randn(512, 512))
        self.b16_conv1_affine_bias = torch.nn.Parameter(torch.full([512], 1.0)) 

        self.b16_conv1_weight = torch.nn.Parameter(torch.randn(512, 512, 3, 3).to(memory_format=torch.contiguous_format))
        self.register_buffer('b16_conv1_noise_const', torch.randn(16, 16))
        self.b16_conv1_noise_strength = torch.nn.Parameter(torch.zeros([]))
        self.b16_conv1_bias = torch.nn.Parameter(torch.zeros([512]))
        # self.b16_torgb = ToRGBLayer(512, 3, w_dim=512,conv_clamp=256, channels_last=False)
        # self.b16_torgb_affine = FullyConnectedLayer(512, 512, bias_init=1)
        self.b16_torgb_affine_weight = torch.nn.Parameter(torch.randn(512, 512))
        self.b16_torgb_affine_bias = torch.nn.Parameter(torch.full([512], 1.0)) 

        self.b16_torgb_weight = torch.nn.Parameter(torch.randn(3, 512, 1, 1).to(memory_format=torch.contiguous_format))
        self.b16_torgb_bias = torch.nn.Parameter(torch.zeros([3]))
        self.b16_torgb_weight_gain = 1 / np.sqrt(512)
        self.b16_num_conv=2
        self.b16_num_torgb=1

        # b32
        self.b32_in_channels = 512
        self.b32_resolution = 32
        self.b32_is_last = False
        self.b32_use_fp16 = False
        #self.b32_conv0 = SynthesisLayer(512, 512, w_dim=512, resolution=32, up=2,resample_filter=[1,3,3,1], conv_clamp=256, channels_last=False)
        self.b32_conv0_resolution = 32
        self.b32_conv0_up = 2
        self.b32_conv0_padding = 3//2
        # self.b32_conv0_affine = FullyConnectedLayer(512, 512, bias_init=1)
        self.b32_conv0_affine_weight = torch.nn.Parameter(torch.randn(512, 512))
        self.b32_conv0_affine_bias = torch.nn.Parameter(torch.full([512], 1.0)) 

        self.b32_conv0_weight = torch.nn.Parameter(torch.randn(512, 512, 3, 3).to(memory_format=torch.contiguous_format))
        self.register_buffer('b32_conv0_noise_const', torch.randn(32, 32))
        self.b32_conv0_noise_strength = torch.nn.Parameter(torch.zeros([]))
        self.b32_conv0_bias = torch.nn.Parameter(torch.zeros([512]))
        #self.b32_conv1 = SynthesisLayer(512, 512, w_dim=512, resolution=32,conv_clamp=256, channels_last=False)
        self.b32_conv1_resolution = 32
        self.b32_conv1_up = 1
        self.b32_conv1_padding = 3//2
        # self.b32_conv1_affine = FullyConnectedLayer(512, 512, bias_init=1)
        self.b32_conv1_affine_weight = torch.nn.Parameter(torch.randn(512, 512))
        self.b32_conv1_affine_bias = torch.nn.Parameter(torch.full([512], 1.0)) 

        self.b32_conv1_weight = torch.nn.Parameter(torch.randn(512, 512, 3, 3).to(memory_format=torch.contiguous_format))
        self.register_buffer('b32_conv1_noise_const', torch.randn(32, 32))
        self.b32_conv1_noise_strength = torch.nn.Parameter(torch.zeros([]))
        self.b32_conv1_bias = torch.nn.Parameter(torch.zeros([512]))
        # self.b32_torgb = ToRGBLayer(512, 3, w_dim=512,conv_clamp=256, channels_last=False)
        # self.b32_torgb_affine = FullyConnectedLayer(512, 512, bias_init=1)
        self.b32_torgb_affine_weight = torch.nn.Parameter(torch.randn(512, 512))
        self.b32_torgb_affine_bias = torch.nn.Parameter(torch.full([512], 1.0))

        self.b32_torgb_weight = torch.nn.Parameter(torch.randn(3, 512, 1, 1).to(memory_format=torch.contiguous_format))
        self.b32_torgb_bias = torch.nn.Parameter(torch.zeros([3]))
        self.b32_torgb_weight_gain = 1 / np.sqrt(512)
        self.b32_num_conv=2
        self.b32_num_torgb=1
        
        # b64
        self.b64_in_channels = 512
        self.b64_resolution = 64
        self.b64_is_last = False
        self.b64_use_fp16 = False
        #self.b64_conv0 = SynthesisLayer(512, 512, w_dim=512, resolution=64, up=2,resample_filter=[1,3,3,1], conv_clamp=256, channels_last=False)
        self.b64_conv0_resolution = 64
        self.b64_conv0_up = 2
        self.b64_conv0_padding = 3//2
        # self.b64_conv0_affine = FullyConnectedLayer(512, 512, bias_init=1)
        self.b64_conv0_affine_weight = torch.nn.Parameter(torch.randn(512, 512))
        self.b64_conv0_affine_bias = torch.nn.Parameter(torch.full([512], 1.0)) 

        self.b64_conv0_weight = torch.nn.Parameter(torch.randn(512, 512, 3, 3).to(memory_format=torch.contiguous_format))
        self.register_buffer('b64_conv0_noise_const', torch.randn(64, 64))
        self.b64_conv0_noise_strength = torch.nn.Parameter(torch.zeros([]))
        self.b64_conv0_bias = torch.nn.Parameter(torch.zeros([512]))
        #self.b64_conv1 = SynthesisLayer(512, 512, w_dim=512, resolution=64,conv_clamp=256, channels_last=False)
        self.b64_conv1_resolution = 64
        self.b64_conv1_up = 1
        self.b64_conv1_padding = 3//2
        # self.b64_conv1_affine = FullyConnectedLayer(512, 512, bias_init=1)
        self.b64_conv1_affine_weight = torch.nn.Parameter(torch.randn(512, 512))
        self.b64_conv1_affine_bias = torch.nn.Parameter(torch.full([512], 1.0)) 

        self.b64_conv1_weight = torch.nn.Parameter(torch.randn(512, 512, 3, 3).to(memory_format=torch.contiguous_format))
        self.register_buffer('b64_conv1_noise_const', torch.randn(64, 64))
        self.b64_conv1_noise_strength = torch.nn.Parameter(torch.zeros([]))
        self.b64_conv1_bias = torch.nn.Parameter(torch.zeros([512]))
        # self.b64_torgb = ToRGBLayer(512, 3, w_dim=512,conv_clamp=256, channels_last=False)
        # self.b64_torgb_affine = FullyConnectedLayer(512, 512, bias_init=1)
        self.b64_torgb_affine_weight = torch.nn.Parameter(torch.randn(512, 512))
        self.b64_torgb_affine_bias = torch.nn.Parameter(torch.full([512], 1.0)) 

        self.b64_torgb_weight = torch.nn.Parameter(torch.randn(3, 512, 1, 1).to(memory_format=torch.contiguous_format))
        self.b64_torgb_bias = torch.nn.Parameter(torch.zeros([3]))
        self.b64_torgb_weight_gain = 1 / np.sqrt(512)
        self.b64_num_conv=2
        self.b64_num_torgb=1
        
        # b128
        self.b128_in_channels = 512
        self.b128_resolution = 128
        self.b128_is_last = False
        self.b128_use_fp16 = True
        #self.b128_conv0 = SynthesisLayer(512, 256, w_dim=512, resolution=128, up=2,resample_filter=[1,3,3,1], conv_clamp=256, channels_last=False)
        self.b128_conv0_resolution = 128
        self.b128_conv0_up = 2
        self.b128_conv0_padding = 3//2
        # self.b128_conv0_affine = FullyConnectedLayer(512, 512, bias_init=1)
        self.b128_conv0_affine_weight = torch.nn.Parameter(torch.randn(512, 512))
        self.b128_conv0_affine_bias = torch.nn.Parameter(torch.full([512], 1.0)) 

        self.b128_conv0_weight = torch.nn.Parameter(torch.randn(256, 512, 3, 3).to(memory_format=torch.contiguous_format))
        self.register_buffer('b128_conv0_noise_const', torch.randn(128, 128))
        self.b128_conv0_noise_strength = torch.nn.Parameter(torch.zeros([]))
        self.b128_conv0_bias = torch.nn.Parameter(torch.zeros([256]))
        #self.b128_conv1 = SynthesisLayer(256, 256, w_dim=512, resolution=128,conv_clamp=256, channels_last=False)
        self.b128_conv1_resolution = 128
        self.b128_conv1_up = 1
        self.b128_conv1_padding = 3//2
        # self.b128_conv1_affine = FullyConnectedLayer(512, 256, bias_init=1)
        self.b128_conv1_affine_weight = torch.nn.Parameter(torch.randn(256, 512))
        self.b128_conv1_affine_bias = torch.nn.Parameter(torch.full([256], 1.0)) 

        self.b128_conv1_weight = torch.nn.Parameter(torch.randn(256, 256, 3, 3).to(memory_format=torch.contiguous_format))
        self.register_buffer('b128_conv1_noise_const', torch.randn(128, 128))
        self.b128_conv1_noise_strength = torch.nn.Parameter(torch.zeros([]))
        self.b128_conv1_bias = torch.nn.Parameter(torch.zeros([256]))
        # self.b128_torgb = ToRGBLayer(256, 3, w_dim=512,conv_clamp=256, channels_last=False)
        # self.b128_torgb_affine = FullyConnectedLayer(512, 256, bias_init=1)
        self.b128_torgb_affine_weight = torch.nn.Parameter(torch.randn(256, 512))
        self.b128_torgb_affine_bias = torch.nn.Parameter(torch.full([256], 1.0)) 

        self.b128_torgb_weight = torch.nn.Parameter(torch.randn(3, 256, 1, 1).to(memory_format=torch.contiguous_format))
        self.b128_torgb_bias = torch.nn.Parameter(torch.zeros([3]))
        self.b128_torgb_weight_gain = 1 / np.sqrt(256)
        self.b128_num_conv=2
        self.b128_num_torgb=1
        
        # b256
        self.b256_in_channels = 256
        self.b256_resolution = 256
        self.b256_is_last = False
        self.b256_use_fp16 = True
        #self.b256_conv0 = SynthesisLayer(256, 128, w_dim=512, resolution=256, up=2,resample_filter=[1,3,3,1], conv_clamp=256, channels_last=False)
        self.b256_conv0_resolution = 256
        self.b256_conv0_up = 2
        self.b256_conv0_padding = 3//2
        # self.b256_conv0_affine = FullyConnectedLayer(512, 256, bias_init=1)
        self.b256_conv0_affine_weight = torch.nn.Parameter(torch.randn(256, 512))
        self.b256_conv0_affine_bias = torch.nn.Parameter(torch.full([256], 1.0)) 

        self.b256_conv0_weight = torch.nn.Parameter(torch.randn(128, 256, 3, 3).to(memory_format=torch.contiguous_format))
        self.register_buffer('b256_conv0_noise_const', torch.randn(256, 256))
        self.b256_conv0_noise_strength = torch.nn.Parameter(torch.zeros([]))
        self.b256_conv0_bias = torch.nn.Parameter(torch.zeros([128]))
        #self.b256_conv1 = SynthesisLayer(128, 128, w_dim=512, resolution=256,conv_clamp=256, channels_last=False)
        self.b256_conv1_resolution = 256
        self.b256_conv1_up = 1
        self.b256_conv1_padding = 3//2
        # self.b256_conv1_affine = FullyConnectedLayer(512, 128, bias_init=1)
        self.b256_conv1_affine_weight = torch.nn.Parameter(torch.randn(128, 512))
        self.b256_conv1_affine_bias = torch.nn.Parameter(torch.full([128], 1.0)) 

        self.b256_conv1_weight = torch.nn.Parameter(torch.randn(128, 128, 3, 3).to(memory_format=torch.contiguous_format))
        self.register_buffer('b256_conv1_noise_const', torch.randn(256, 256))
        self.b256_conv1_noise_strength = torch.nn.Parameter(torch.zeros([]))
        self.b256_conv1_bias = torch.nn.Parameter(torch.zeros([128]))
        # self.b256_torgb = ToRGBLayer(128, 3, w_dim=512,conv_clamp=256, channels_last=False)
        # self.b256_torgb_affine = FullyConnectedLayer(512, 128, bias_init=1)
        self.b256_torgb_affine_weight = torch.nn.Parameter(torch.randn(128, 512))
        self.b256_torgb_affine_bias = torch.nn.Parameter(torch.full([128], 1.0)) 

        self.b256_torgb_weight = torch.nn.Parameter(torch.randn(3, 128, 1, 1).to(memory_format=torch.contiguous_format))
        self.b256_torgb_bias = torch.nn.Parameter(torch.zeros([3]))
        self.b256_torgb_weight_gain = 1 / np.sqrt(128)
        self.b256_num_conv=2
        self.b256_num_torgb=1
        
        # b512
        self.b512_in_channels = 128
        self.b512_resolution = 512
        self.b512_is_last = False
        self.b512_use_fp16 = True
        #self.b512_conv0 = SynthesisLayer(128, 64, w_dim=512, resolution=512, up=2,resample_filter=[1,3,3,1], conv_clamp=256, channels_last=False)
        self.b512_conv0_resolution = 512
        self.b512_conv0_up = 2
        self.b512_conv0_padding = 3//2
        # self.b512_conv0_affine = FullyConnectedLayer(512, 128, bias_init=1)
        self.b512_conv0_affine_weight = torch.nn.Parameter(torch.randn(128, 512))
        self.b512_conv0_affine_bias = torch.nn.Parameter(torch.full([128], 1.0)) 

        self.b512_conv0_weight = torch.nn.Parameter(torch.randn(64, 128, 3, 3).to(memory_format=torch.contiguous_format))
        self.register_buffer('b512_conv0_noise_const', torch.randn(512, 512))
        self.b512_conv0_noise_strength = torch.nn.Parameter(torch.zeros([]))
        self.b512_conv0_bias = torch.nn.Parameter(torch.zeros([64]))
        #self.b512_conv1 = SynthesisLayer(64, 64, w_dim=512, resolution=512,conv_clamp=256, channels_last=False)
        self.b512_conv1_resolution = 512
        self.b512_conv1_up = 1
        self.b512_conv1_padding = 3//2
        # self.b512_conv1_affine = FullyConnectedLayer(512, 64, bias_init=1)
        self.b512_conv1_affine_weight = torch.nn.Parameter(torch.randn(64, 512))
        self.b512_conv1_affine_bias = torch.nn.Parameter(torch.full([64], 1.0)) 

        self.b512_conv1_weight = torch.nn.Parameter(torch.randn(64, 64, 3, 3).to(memory_format=torch.contiguous_format))
        self.register_buffer('b512_conv1_noise_const', torch.randn(512, 512))
        self.b512_conv1_noise_strength = torch.nn.Parameter(torch.zeros([]))
        self.b512_conv1_bias = torch.nn.Parameter(torch.zeros([64]))
        # self.b512_torgb = ToRGBLayer(64, 3, w_dim=512,conv_clamp=256, channels_last=False)
        # self.b512_torgb_affine = FullyConnectedLayer(512, 64, bias_init=1)
        self.b512_torgb_affine_weight = torch.nn.Parameter(torch.randn(64, 512))
        self.b512_torgb_affine_bias = torch.nn.Parameter(torch.full([64], 1.0)) 

        self.b512_torgb_weight = torch.nn.Parameter(torch.randn(3, 64, 1, 1).to(memory_format=torch.contiguous_format))
        self.b512_torgb_bias = torch.nn.Parameter(torch.zeros([3]))
        self.b512_torgb_weight_gain = 1 / np.sqrt(64)
        self.b512_num_conv=2
        self.b512_num_torgb=1
        
        # b1024
        self.b1024_in_channels = 64
        self.b1024_resolution = 1024
        self.b1024_is_last = True
        self.b1024_use_fp16 = True
        #self.b1024_conv0 = SynthesisLayer(64, 32, w_dim=512, resolution=1024, up=2,resample_filter=[1,3,3,1], conv_clamp=256, channels_last=False)
        self.b1024_conv0_resolution = 1024
        self.b1024_conv0_up = 2
        self.b1024_conv0_padding = 3//2
        # self.b1024_conv0_affine = FullyConnectedLayer(512, 64, bias_init=1)
        self.b1024_conv0_affine_weight = torch.nn.Parameter(torch.randn(64, 512))
        self.b1024_conv0_affine_bias = torch.nn.Parameter(torch.full([64], 1.0)) 

        self.b1024_conv0_weight = torch.nn.Parameter(torch.randn(32, 64, 3, 3).to(memory_format=torch.contiguous_format))
        self.register_buffer('b1024_conv0_noise_const', torch.randn(1024, 1024))
        self.b1024_conv0_noise_strength = torch.nn.Parameter(torch.zeros([]))
        self.b1024_conv0_bias = torch.nn.Parameter(torch.zeros([32]))
        #self.b1024_conv1 = SynthesisLayer(32, 32, w_dim=512, resolution=1024,conv_clamp=256, channels_last=False)
        self.b1024_conv1_resolution = 1024
        self.b1024_conv1_up = 1
        self.b1024_conv1_padding = 3//2
        # self.b1024_conv1_affine = FullyConnectedLayer(512, 32, bias_init=1)
        self.b1024_conv1_affine_weight = torch.nn.Parameter(torch.randn(32, 512))
        self.b1024_conv1_affine_bias = torch.nn.Parameter(torch.full([32], 1.0)) 

        self.b1024_conv1_weight = torch.nn.Parameter(torch.randn(32, 32, 3, 3).to(memory_format=torch.contiguous_format))
        self.register_buffer('b1024_conv1_noise_const', torch.randn(1024, 1024))
        self.b1024_conv1_noise_strength = torch.nn.Parameter(torch.zeros([]))
        self.b1024_conv1_bias = torch.nn.Parameter(torch.zeros([32]))
        # self.b1024_torgb = ToRGBLayer(32, 3, w_dim=512,conv_clamp=256, channels_last=False)
        # self.b1024_torgb_affine = FullyConnectedLayer(512, 32, bias_init=1)
        self.b1024_torgb_affine_weight = torch.nn.Parameter(torch.randn(32, 512))
        self.b1024_torgb_affine_bias = torch.nn.Parameter(torch.full([32], 1.0)) 

        self.b1024_torgb_weight = torch.nn.Parameter(torch.randn(3, 32, 1, 1).to(memory_format=torch.contiguous_format))
        self.b1024_torgb_bias = torch.nn.Parameter(torch.zeros([3]))
        self.b1024_torgb_weight_gain = 1 / np.sqrt(32)
        self.b1024_num_conv=2
        self.b1024_num_torgb=1
    def forward(self, ws, force_fp32=False, noise_mode='random'):
        # dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        # fused_modconv = (not self.training) and (dtype == torch.float32 or int(x.shape[0]) == 1)
        dtype32=torch.float32
        dtype16=torch.float16 if not force_fp32 else torch.float32
        contiguous=torch.contiguous_format
        # w_iter = iter(ws.unbind(dim=1))

        fused_modconv =not self.training
        N=ws.shape[0]
        ############################################################################################################################
        # b4 :{'input_x':None,'input_img':None,'output_x':[1, 512, 4, 4],'output_img':[1, 3, 4, 4]},
        ############################################################################################################################
        x=self.b4_const.unsqueeze(0).repeat([N, 1, 1, 1])
        # print("b4_contiguous",x.is_contiguous())
            # x = self.b4_conv1(self.b4_const, ws[:,0,:], fused_modconv=fused_modconv, noise_mode=noise_mode)
        # misc.assert_shape(x, [None, 512, 4, 4])
        noise = torch.randn([N, 1, 4, 4], device=x.device) * self.b4_conv1_noise_strength if noise_mode == 'random' else self.b4_conv1_noise_const * self.b4_conv1_noise_strength
        x = modulated_conv2d(x=x, weight=self.b4_conv1_weight, 
            # styles=self.b4_conv1_affine(ws[:,0,:]), 
            styles=torch.addmm(self.b4_conv1_affine_bias.to(ws.dtype).unsqueeze(0), ws[:,0,:], (self.b4_conv1_affine_weight.to(ws.dtype) * 0.044194173824159216).t()),
            noise=noise, up=1, flip_weight=True,padding=1, resample_filter=self.resample_filter, fused_modconv=fused_modconv)
        x = bias_act.bias_act(x, self.b4_conv1_bias.to(x.dtype), gain=np.sqrt(2), act='lrelu', clamp=256)

            # img = self.b4_torgb(x, ws[:,1,:], fused_modconv=fused_modconv)
        _x = modulated_conv2d(x=x, weight=self.b4_torgb_weight, 
            # styles=self.b4_torgb_affine(ws[:,1,:]) * self.b4_torgb_weight_gain, 
            styles=torch.addmm(self.b4_torgb_affine_bias.to(ws.dtype).unsqueeze(0), ws[:,1,:], (self.b4_torgb_affine_weight.to(ws.dtype) * 0.044194173824159216).t())* self.b4_torgb_weight_gain,
            demodulate=False, fused_modconv=fused_modconv)
        img = bias_act.bias_act(_x, self.b4_torgb_bias.to(x.dtype), clamp=256)
        img = img.to(dtype=dtype32, memory_format=contiguous)
        ############################################################################################################################
        # b8 :{'input_x':[1, 512, 4, 4],'input_img':[1, 3, 4, 4],'output_x':[1, 512, 8, 8],'output_img':[1, 3, 8, 8]},
        ############################################################################################################################
        # print("b8_contiguous",x.is_contiguous())
        x = x.to(dtype=dtype32,memory_format=contiguous)
            # x = self.b8_conv0(x,  ws[:,1,:], fused_modconv=fused_modconv, noise_mode=noise_mode)
        # misc.assert_shape(x, [None, 512, 4, 4])
        noise = torch.randn([N, 1, 8, 8], device=x.device) * self.b8_conv0_noise_strength if noise_mode == 'random' else self.b8_conv0_noise_const * self.b8_conv0_noise_strength
        x = modulated_conv2d(x=x, weight=self.b8_conv0_weight, 
            # styles=self.b8_conv0_affine(ws[:,1,:]), 
            styles=torch.addmm(self.b8_conv0_affine_bias.to(ws.dtype).unsqueeze(0), ws[:,1,:], (self.b8_conv0_affine_weight.to(ws.dtype) * 0.044194173824159216).t()),
            noise=noise, up=2, flip_weight=False,padding=1, resample_filter=self.resample_filter, fused_modconv=fused_modconv)
        x = bias_act.bias_act(x, self.b8_conv0_bias.to(x.dtype), gain=np.sqrt(2), act='lrelu', clamp=256)
            # x = self.b8_conv1(x,  ws[:,2,:], fused_modconv=fused_modconv, noise_mode=noise_mode)
        # misc.assert_shape(x, [None, 512, 8, 8])
        noise = torch.randn([N, 1, 8, 8], device=x.device) * self.b8_conv1_noise_strength if noise_mode == 'random' else self.b8_conv1_noise_const * self.b8_conv1_noise_strength
        x = modulated_conv2d(x=x, weight=self.b8_conv1_weight, 
            # styles=self.b8_conv1_affine(ws[:,2,:]), 
            styles=torch.addmm(self.b8_conv1_affine_bias.to(ws.dtype).unsqueeze(0), ws[:,2,:], (self.b8_conv1_affine_weight.to(ws.dtype) * 0.044194173824159216).t()),
            noise=noise, up=1, flip_weight=True,padding=1, resample_filter=self.resample_filter, fused_modconv=fused_modconv)
        x = bias_act.bias_act(x, self.b8_conv1_bias.to(x.dtype), gain=np.sqrt(2), act='lrelu', clamp=256)

        img = upfirdn2d.upsample2d(img, self.resample_filter)
            #y = self.b8_torgb(x,  ws[:,3,:], fused_modconv=fused_modconv)
        _x = modulated_conv2d(x=x, weight=self.b8_torgb_weight, 
            # styles=self.b8_torgb_affine(ws[:,3,:]) * self.b8_torgb_weight_gain, 
            styles=torch.addmm(self.b8_torgb_affine_bias.to(ws.dtype).unsqueeze(0), ws[:,3,:], (self.b8_torgb_affine_weight.to(ws.dtype) * 0.044194173824159216).t())* self.b8_torgb_weight_gain,
            demodulate=False, fused_modconv=fused_modconv)
        y = bias_act.bias_act(_x, self.b8_torgb_bias.to(x.dtype), clamp=256)
        y = y.to(dtype=dtype32, memory_format=contiguous)
        img = img.add_(y) 
        ############################################################################################################################
        # b16 :{'input_x':[1, 512, 8, 8],'input_img':[1, 3, 8, 8],'output_x':[1, 512, 16,16],'output_img':[1, 3, 16, 16]},
        ############################################################################################################################
        # print("b16_contiguous",x.is_contiguous())
        x = x.to(dtype=dtype32,memory_format=contiguous)
            # x = self.b16_conv0(x,  ws[:,3,:], fused_modconv=fused_modconv, noise_mode=noise_mode)
        # misc.assert_shape(x, [None, 512, 8, 8])
        noise = torch.randn([N, 1, 16, 16], device=x.device) * self.b16_conv0_noise_strength if noise_mode == 'random' else self.b16_conv0_noise_const * self.b16_conv0_noise_strength
        x = modulated_conv2d(x=x, weight=self.b16_conv0_weight, 
            # styles=self.b16_conv0_affine(ws[:,3,:]), 
            styles=torch.addmm(self.b16_conv0_affine_bias.to(ws.dtype).unsqueeze(0), ws[:,3,:], (self.b16_conv0_affine_weight.to(ws.dtype) * 0.044194173824159216).t()),
            noise=noise, up=2, flip_weight=False,padding=1, resample_filter=self.resample_filter, fused_modconv=fused_modconv)
        x = bias_act.bias_act(x, self.b16_conv0_bias.to(x.dtype), gain=np.sqrt(2), act='lrelu', clamp=256)
            # x = self.b16_conv1(x,  ws[:,4,:], fused_modconv=fused_modconv, noise_mode=noise_mode)
        # misc.assert_shape(x, [None, 512, 16, 16])
        noise = torch.randn([N, 1, 16, 16], device=x.device) * self.b16_conv1_noise_strength if noise_mode == 'random' else self.b16_conv1_noise_const * self.b16_conv1_noise_strength
        x = modulated_conv2d(x=x, weight=self.b16_conv1_weight, 
            # styles=self.b16_conv1_affine(ws[:,4,:]), 
            styles=torch.addmm(self.b16_conv1_affine_bias.to(ws.dtype).unsqueeze(0), ws[:,4,:], (self.b16_conv1_affine_weight.to(ws.dtype) * 0.044194173824159216).t()),
            noise=noise, up=1, flip_weight=True,padding=1, resample_filter=self.resample_filter, fused_modconv=fused_modconv)
        x = bias_act.bias_act(x, self.b16_conv1_bias.to(x.dtype), gain=np.sqrt(2), act='lrelu', clamp=256)

        img = upfirdn2d.upsample2d(img, self.resample_filter)
            #y = self.b16_torgb(x,  ws[:,5,:], fused_modconv=fused_modconv)
        _x = modulated_conv2d(x=x, weight=self.b16_torgb_weight, 
            # styles=self.b16_torgb_affine(ws[:,5,:]) * self.b16_torgb_weight_gain, 
            styles=torch.addmm(self.b16_torgb_affine_bias.to(ws.dtype).unsqueeze(0), ws[:,5,:], (self.b16_torgb_affine_weight.to(ws.dtype) * 0.044194173824159216).t())* self.b16_torgb_weight_gain, 
            demodulate=False, fused_modconv=fused_modconv)
        y = bias_act.bias_act(_x, self.b16_torgb_bias.to(x.dtype), clamp=256)
        y = y.to(dtype=dtype32, memory_format=contiguous)
        img = img.add_(y) 
        ############################################################################################################################
        # b32 :{'input_x':[1, 512, 16,16],'input_img':[1, 3, 16, 16],'output_x':[1, 512, 32,32],'output_img':[1, 3, 32, 32]},
        ############################################################################################################################
        # print("b32_contiguous",x.is_contiguous())
        x = x.to(dtype=dtype32,memory_format=contiguous)
            # x = self.b32_conv0(x,  ws[:,5,:], fused_modconv=fused_modconv, noise_mode=noise_mode)
        # misc.assert_shape(x, [None, 512, 16, 16])
        noise = torch.randn([N, 1, 32, 32], device=x.device) * self.b32_conv0_noise_strength if noise_mode == 'random' else self.b32_conv0_noise_const * self.b32_conv0_noise_strength
        x = modulated_conv2d(x=x, weight=self.b32_conv0_weight, 
            # styles=self.b32_conv0_affine(ws[:,5,:]), 
            styles=torch.addmm(self.b32_conv0_affine_bias.to(ws.dtype).unsqueeze(0), ws[:,5,:], (self.b32_conv0_affine_weight.to(ws.dtype) * 0.044194173824159216).t()),
            noise=noise, up=2, flip_weight=False,padding=1, resample_filter=self.resample_filter, fused_modconv=fused_modconv)
        x = bias_act.bias_act(x, self.b32_conv0_bias.to(x.dtype), gain=np.sqrt(2), act='lrelu', clamp=256)
            # x = self.b32_conv1(x,  ws[:,6,:], fused_modconv=fused_modconv, noise_mode=noise_mode)
        # misc.assert_shape(x, [None, 512, 32, 32])
        noise = torch.randn([N, 1, 32, 32], device=x.device) * self.b32_conv1_noise_strength if noise_mode == 'random' else self.b32_conv1_noise_const * self.b32_conv1_noise_strength
        x = modulated_conv2d(x=x, weight=self.b32_conv1_weight, 
            # styles=self.b32_conv1_affine(ws[:,6,:]), 
            styles=torch.addmm(self.b32_conv1_affine_bias.to(ws.dtype).unsqueeze(0), ws[:,6,:], (self.b32_conv1_affine_weight.to(ws.dtype) * 0.044194173824159216).t()),
            noise=noise, up=1, flip_weight=True,padding=1, resample_filter=self.resample_filter, fused_modconv=fused_modconv)
        x = bias_act.bias_act(x, self.b32_conv1_bias.to(x.dtype), gain=np.sqrt(2), act='lrelu', clamp=256)

        img = upfirdn2d.upsample2d(img, self.resample_filter)
            #y = self.b32_torgb(x,  ws[:,7,:], fused_modconv=fused_modconv)
        _x = modulated_conv2d(x=x, weight=self.b32_torgb_weight, 
            # styles=self.b32_torgb_affine(ws[:,7,:]) * self.b32_torgb_weight_gain, 
            styles=torch.addmm(self.b32_torgb_affine_bias.to(ws.dtype).unsqueeze(0), ws[:,7,:], (self.b32_torgb_affine_weight.to(ws.dtype) * 0.044194173824159216).t())* self.b32_torgb_weight_gain, 
            demodulate=False, fused_modconv=fused_modconv)
        y = bias_act.bias_act(_x, self.b32_torgb_bias.to(x.dtype), clamp=256)
        y = y.to(dtype=dtype32, memory_format=contiguous)
        img = img.add_(y) 
        ############################################################################################################################
        # b64 :{'input_x':[1, 512, 32,32],'input_img':[1, 3, 32, 32],'output_x':[1, 512, 64,64],'output_img':[1, 3, 64, 64]},
        ############################################################################################################################
        # print("b64_contiguous",x.is_contiguous())
        x = x.to(dtype=dtype32,memory_format=contiguous)
            # x = self.b64_conv0(x,  ws[:,7,:], fused_modconv=fused_modconv, noise_mode=noise_mode)
        # misc.assert_shape(x, [None, 512, 32, 32])
        noise = torch.randn([N, 1, 64, 64], device=x.device) * self.b64_conv0_noise_strength if noise_mode == 'random' else self.b64_conv0_noise_const * self.b64_conv0_noise_strength
        x = modulated_conv2d(x=x, weight=self.b64_conv0_weight, 
            # styles=self.b64_conv0_affine(ws[:,7,:]),
            styles=torch.addmm(self.b64_conv0_affine_bias.to(ws.dtype).unsqueeze(0), ws[:,7,:], (self.b64_conv0_affine_weight.to(ws.dtype) * 0.044194173824159216).t()), 
            noise=noise, up=2, flip_weight=False,padding=1, resample_filter=self.resample_filter, fused_modconv=fused_modconv)
        x = bias_act.bias_act(x, self.b64_conv0_bias.to(x.dtype), gain=np.sqrt(2), act='lrelu', clamp=256)
            # x = self.b64_conv1(x,  ws[:,8,:], fused_modconv=fused_modconv, noise_mode=noise_mode)
        # misc.assert_shape(x, [None, 512, 64, 64])
        noise = torch.randn([N, 1, 64, 64], device=x.device) * self.b64_conv1_noise_strength if noise_mode == 'random' else self.b64_conv1_noise_const * self.b64_conv1_noise_strength
        x = modulated_conv2d(x=x, weight=self.b64_conv1_weight, 
            # styles=self.b64_conv1_affine(ws[:,8,:]), 
            styles=torch.addmm(self.b64_conv1_affine_bias.to(ws.dtype).unsqueeze(0), ws[:,8,:], (self.b64_conv1_affine_weight.to(ws.dtype) * 0.044194173824159216).t()),
            noise=noise, up=1, flip_weight=True,padding=1, resample_filter=self.resample_filter, fused_modconv=fused_modconv)
        x = bias_act.bias_act(x, self.b64_conv1_bias.to(x.dtype), gain=np.sqrt(2), act='lrelu', clamp=256)

        img = upfirdn2d.upsample2d(img, self.resample_filter)
            #y = self.b64_torgb(x,  ws[:,9,:], fused_modconv=fused_modconv)
        _x = modulated_conv2d(x=x, weight=self.b64_torgb_weight, 
            # styles=self.b64_torgb_affine(ws[:,9,:]) * self.b64_torgb_weight_gain, 
            styles=torch.addmm(self.b64_torgb_affine_bias.to(ws.dtype).unsqueeze(0), ws[:,9,:], (self.b64_torgb_affine_weight.to(ws.dtype) * 0.044194173824159216).t()) * self.b64_torgb_weight_gain,
            demodulate=False, fused_modconv=fused_modconv)
        y = bias_act.bias_act(_x, self.b64_torgb_bias.to(x.dtype), clamp=256)
        y = y.to(dtype=dtype32, memory_format=contiguous)
        img = img.add_(y) 

        # 以下层可以使用fp16, fused_modconv相应修改
        ############################################################################################################################
        # b128 :{'input_x':[1, 512, 64,64],'input_img':[1, 3, 64, 64],'output_x':[1, 256, 128,128],'output_img':[1, 3, 128, 128]},
        ############################################################################################################################
        # print("b128_contiguous",x.is_contiguous())
        fused_modconv = (not self.training) and (int(N) == 1)

        x = x.to(dtype=dtype16,memory_format=contiguous)
            # x = self.b128_conv0(x,  ws[:,9,:], fused_modconv=fused_modconv, noise_mode=noise_mode)
        # misc.assert_shape(x, [None, 512, 64, 64])
        noise = torch.randn([N, 1, 128, 128], device=x.device) * self.b128_conv0_noise_strength if noise_mode == 'random' else self.b128_conv0_noise_const * self.b128_conv0_noise_strength
        x = modulated_conv2d(x=x, weight=self.b128_conv0_weight, 
            # styles=self.b128_conv0_affine(ws[:,9,:]), 
            styles=torch.addmm(self.b128_conv0_affine_bias.to(ws.dtype).unsqueeze(0), ws[:,9,:], (self.b128_conv0_affine_weight.to(ws.dtype) * 0.044194173824159216).t()),
            noise=noise, up=2, flip_weight=False,padding=1, resample_filter=self.resample_filter, fused_modconv=fused_modconv)
        x = bias_act.bias_act(x, self.b128_conv0_bias.to(x.dtype), gain=np.sqrt(2), act='lrelu', clamp=256)
            # x = self.b128_conv1(x,  ws[:,10,:], fused_modconv=fused_modconv, noise_mode=noise_mode)
        # misc.assert_shape(x, [None, 256, 128, 128])
        noise = torch.randn([N, 1, 128, 128], device=x.device) * self.b128_conv1_noise_strength if noise_mode == 'random' else self.b128_conv1_noise_const * self.b128_conv1_noise_strength
        x = modulated_conv2d(x=x, weight=self.b128_conv1_weight, 
            # styles=self.b128_conv1_affine(ws[:,10,:]), 
            styles=torch.addmm(self.b128_conv1_affine_bias.to(ws.dtype).unsqueeze(0), ws[:,10,:], (self.b128_conv1_affine_weight.to(ws.dtype) * 0.044194173824159216).t()),
            noise=noise, up=1, flip_weight=True,padding=1, resample_filter=self.resample_filter, fused_modconv=fused_modconv)
        x = bias_act.bias_act(x, self.b128_conv1_bias.to(x.dtype), gain=np.sqrt(2), act='lrelu', clamp=256)

        img = upfirdn2d.upsample2d(img, self.resample_filter)
            #y = self.b128_torgb(x,  ws[:,11,:], fused_modconv=fused_modconv)
        _x = modulated_conv2d(x=x, weight=self.b128_torgb_weight, 
            # styles=self.b128_torgb_affine(ws[:,11,:]) * self.b128_torgb_weight_gain, 
            styles=torch.addmm(self.b128_torgb_affine_bias.to(ws.dtype).unsqueeze(0), ws[:,11,:], (self.b128_torgb_affine_weight.to(ws.dtype) * 0.044194173824159216).t()) * self.b128_torgb_weight_gain, 
            demodulate=False, fused_modconv=fused_modconv)
        y = bias_act.bias_act(_x, self.b128_torgb_bias.to(x.dtype), clamp=256)
        y = y.to(dtype=dtype32, memory_format=contiguous)
        img = img.add_(y) 
        ############################################################################################################################
        # b256 :{'input_x':[1, 256, 128,128],'input_img':[1, 3, 128, 128],'output_x':[1, 128, 256,256],'output_img':[1, 3, 256, 256]},
        ############################################################################################################################
        # print("b256_contiguous",x.is_contiguous())
        x = x.to(dtype=dtype16,memory_format=contiguous)
            # x = self.b256_conv0(x,  ws[:,11,:], fused_modconv=fused_modconv, noise_mode=noise_mode)
        # misc.assert_shape(x, [None, 256, 128, 128])
        noise = torch.randn([N, 1, 256, 256], device=x.device) * self.b256_conv0_noise_strength if noise_mode == 'random' else self.b256_conv0_noise_const * self.b256_conv0_noise_strength
        x = modulated_conv2d(x=x, weight=self.b256_conv0_weight, 
            # styles=self.b256_conv0_affine(ws[:,11,:]), 
            styles=torch.addmm(self.b256_conv0_affine_bias.to(ws.dtype).unsqueeze(0), ws[:,11,:], (self.b256_conv0_affine_weight.to(ws.dtype) * 0.044194173824159216).t()),
            noise=noise, up=2, flip_weight=False,padding=1, resample_filter=self.resample_filter, fused_modconv=fused_modconv)
        x = bias_act.bias_act(x, self.b256_conv0_bias.to(x.dtype), gain=np.sqrt(2), act='lrelu', clamp=256)
            # x = self.b256_conv1(x,  ws[:,12,:], fused_modconv=fused_modconv, noise_mode=noise_mode)
        # misc.assert_shape(x, [None, 128, 256, 256])
        noise = torch.randn([N, 1, 256, 256], device=x.device) * self.b256_conv1_noise_strength if noise_mode == 'random' else self.b256_conv1_noise_const * self.b256_conv1_noise_strength
        x = modulated_conv2d(x=x, weight=self.b256_conv1_weight, 
            # styles=self.b256_conv1_affine(ws[:,12,:]), 
            styles=torch.addmm(self.b256_conv1_affine_bias.to(ws.dtype).unsqueeze(0), ws[:,12,:], (self.b256_conv1_affine_weight.to(ws.dtype) * 0.044194173824159216).t()),
            noise=noise, up=1, flip_weight=True,padding=1, resample_filter=self.resample_filter, fused_modconv=fused_modconv)
        x = bias_act.bias_act(x, self.b256_conv1_bias.to(x.dtype), gain=np.sqrt(2), act='lrelu', clamp=256)

        img = upfirdn2d.upsample2d(img, self.resample_filter)
            #y = self.b256_torgb(x,  ws[:,13,:], fused_modconv=fused_modconv)
        _x = modulated_conv2d(x=x, weight=self.b256_torgb_weight, 
            # styles=self.b256_torgb_affine(ws[:,13,:]) * self.b256_torgb_weight_gain, 
            styles=torch.addmm(self.b256_torgb_affine_bias.to(ws.dtype).unsqueeze(0), ws[:,13,:], (self.b256_torgb_affine_weight.to(ws.dtype) * 0.044194173824159216).t())* self.b256_torgb_weight_gain, 
            demodulate=False, fused_modconv=fused_modconv)
        y = bias_act.bias_act(_x, self.b256_torgb_bias.to(x.dtype), clamp=256)
        y = y.to(dtype=dtype32, memory_format=contiguous)
        img = img.add_(y) 
        ############################################################################################################################
        # b512 :{'input_x':[1, 128, 256,256],'input_img':[1, 3, 256, 256],'output_x':[1, 64, 512,512],'output_img':[1, 3, 512, 512]},
        ############################################################################################################################
        # print("b512_contiguous",x.is_contiguous())
        x = x.to(dtype=dtype16,memory_format=contiguous)
            # x = self.b512_conv0(x,  ws[:,13,:], fused_modconv=fused_modconv, noise_mode=noise_mode)
        # misc.assert_shape(x, [None, 128, 256, 256])
        noise = torch.randn([N, 1, 512, 512], device=x.device) * self.b512_conv0_noise_strength if noise_mode == 'random' else self.b512_conv0_noise_const * self.b512_conv0_noise_strength
        x = modulated_conv2d(x=x, weight=self.b512_conv0_weight,
            # styles=self.b512_conv0_affine(ws[:,13,:]), 
            styles=torch.addmm(self.b512_conv0_affine_bias.to(ws.dtype).unsqueeze(0), ws[:,13,:], (self.b512_conv0_affine_weight.to(ws.dtype) * 0.044194173824159216).t()),
            noise=noise, up=2, flip_weight=False,padding=1, resample_filter=self.resample_filter, fused_modconv=fused_modconv)
        x = bias_act.bias_act(x, self.b512_conv0_bias.to(x.dtype), gain=np.sqrt(2), act='lrelu', clamp=256)
            # x = self.b512_conv1(x,  ws[:,14,:], fused_modconv=fused_modconv, noise_mode=noise_mode)
        # misc.assert_shape(x, [None, 64, 512, 512])
        noise = torch.randn([N, 1, 512, 512], device=x.device) * self.b512_conv1_noise_strength if noise_mode == 'random' else self.b512_conv1_noise_const * self.b512_conv1_noise_strength
        x = modulated_conv2d(x=x, weight=self.b512_conv1_weight, 
            # styles=self.b512_conv1_affine(ws[:,14,:]), 
            styles=torch.addmm(self.b512_conv1_affine_bias.to(ws.dtype).unsqueeze(0), ws[:,14,:], (self.b512_conv1_affine_weight.to(ws.dtype) * 0.044194173824159216).t()),
            noise=noise, up=1, flip_weight=True,padding=1, resample_filter=self.resample_filter, fused_modconv=fused_modconv)
        x = bias_act.bias_act(x, self.b512_conv1_bias.to(x.dtype), gain=np.sqrt(2), act='lrelu', clamp=256)

        img = upfirdn2d.upsample2d(img, self.resample_filter)
            #y = self.b512_torgb(x,  ws[:,15,:], fused_modconv=fused_modconv)
        _x = modulated_conv2d(x=x, weight=self.b512_torgb_weight, 
            # styles=self.b512_torgb_affine(ws[:,15,:]) * self.b512_torgb_weight_gain, 
            styles=torch.addmm(self.b512_torgb_affine_bias.to(ws.dtype).unsqueeze(0), ws[:,15,:], (self.b512_torgb_affine_weight.to(ws.dtype) * 0.044194173824159216).t()) * self.b512_torgb_weight_gain, 
            demodulate=False, fused_modconv=fused_modconv)
        y = bias_act.bias_act(_x, self.b512_torgb_bias.to(x.dtype), clamp=256)
        y = y.to(dtype=dtype32, memory_format=contiguous)
        img = img.add_(y) 
        ############################################################################################################################
        # b1024 :{'input_x':[1, 64, 512,512],'input_img':[1, 3, 512, 512],'output_x':[1, 32, 1024,1024],'output_img':[1, 3, 1024, 1024]},
        ############################################################################################################################
        # print("b1024_contiguous",x.is_contiguous())
        x = x.to(dtype=dtype16,memory_format=contiguous)
            # x = self.b1024_conv0(x,  ws[:,15,:], fused_modconv=fused_modconv, noise_mode=noise_mode)
        # misc.assert_shape(x, [None, 64, 512, 512])
        noise = torch.randn([N, 1, 1024, 1024], device=x.device) * self.b1024_conv0_noise_strength if noise_mode == 'random' else self.b1024_conv0_noise_const * self.b1024_conv0_noise_strength
        x = modulated_conv2d(x=x, weight=self.b1024_conv0_weight, 
            # styles=self.b1024_conv0_affine(ws[:,15,:]), 
            styles=torch.addmm(self.b1024_conv0_affine_bias.to(ws.dtype).unsqueeze(0), ws[:,15,:], (self.b1024_conv0_affine_weight.to(ws.dtype) * 0.044194173824159216).t()),
            noise=noise, up=2, flip_weight=False,padding=1, resample_filter=self.resample_filter, fused_modconv=fused_modconv)
        x = bias_act.bias_act(x, self.b1024_conv0_bias.to(x.dtype), gain=np.sqrt(2), act='lrelu', clamp=256)
            # x = self.b1024_conv1(x,  ws[:,16,:], fused_modconv=fused_modconv, noise_mode=noise_mode)
        # misc.assert_shape(x, [None, 32, 1024, 1024])
        noise = torch.randn([N, 1, 1024, 1024], device=x.device) * self.b1024_conv1_noise_strength if noise_mode == 'random' else self.b1024_conv1_noise_const * self.b1024_conv1_noise_strength
        x = modulated_conv2d(x=x, weight=self.b1024_conv1_weight, 
            # styles=self.b1024_conv1_affine(ws[:,16,:]), 
            styles=torch.addmm(self.b1024_conv1_affine_bias.to(ws.dtype).unsqueeze(0), ws[:,16,:], (self.b1024_conv1_affine_weight.to(ws.dtype) * 0.044194173824159216).t()),
            noise=noise, up=1, flip_weight=True,padding=1, resample_filter=self.resample_filter, fused_modconv=fused_modconv)
        x = bias_act.bias_act(x, self.b1024_conv1_bias.to(x.dtype), gain=np.sqrt(2), act='lrelu', clamp=256)

        img = upfirdn2d.upsample2d(img, self.resample_filter)
            #y = self.b1024_torgb(x,  ws[:,17,:], fused_modconv=fused_modconv)
        _x = modulated_conv2d(x=x, weight=self.b1024_torgb_weight, 
            # styles=self.b1024_torgb_affine(ws[:,17,:]) * self.b1024_torgb_weight_gain, 
            styles=torch.addmm(self.b1024_torgb_affine_bias.to(ws.dtype).unsqueeze(0), ws[:,17,:], (self.b1024_torgb_affine_weight.to(ws.dtype) * 0.044194173824159216).t())* self.b1024_torgb_weight_gain,
            demodulate=False, fused_modconv=fused_modconv)
        y = bias_act.bias_act(_x, self.b1024_torgb_bias.to(x.dtype), clamp=256)
        y = y.to(dtype=dtype32, memory_format=contiguous)
        img = img.add_(y) 

        # 需要先测试几天网络稳定性，然后再试着删除下列temp数据
        # del _x,x,y,ws
        # torch.cuda.empty_cache()
        return img

    def copy_param_and_buffer(self,synthesis):
        mapping_dict=[
        # Buffer 这是恒定数字，已经在class中定义，但是从peng_pkl反转成official_pkl时，需要把它拷贝到多处，除非official_pkl在class中也有定义
        # {'peng':self.resample_filter,'official':synthesis.b4.resample_filter, 'shape': torch.Size([4, 4]) },

        # b4
        {'peng': self.b4_const , 'official': synthesis.b4.const , 'shape': torch.Size([512, 4, 4]) }, 
        {'peng': self.b4_conv1_affine_weight , 'official': synthesis.b4.conv1.affine.weight , 'shape': torch.Size([512, 512]) }, 
        {'peng': self.b4_conv1_affine_bias , 'official': synthesis.b4.conv1.affine.bias , 'shape': torch.Size([512]) }, 
        {'peng': self.b4_conv1_weight , 'official': synthesis.b4.conv1.weight , 'shape': torch.Size([512, 512, 3, 3]) }, 
        # Buffer
        {'peng': self.b4_conv1_noise_const , 'official':synthesis.b4.conv1.noise_const , 'shape': torch.Size([4, 4]) },

        {'peng': self.b4_conv1_noise_strength , 'official': synthesis.b4.conv1.noise_strength , 'shape': torch.Size([ ]) }, 
        {'peng': self.b4_conv1_bias , 'official': synthesis.b4.conv1.bias , 'shape': torch.Size([512]) }, 
        {'peng': self.b4_torgb_affine_weight , 'official': synthesis.b4.torgb.affine.weight , 'shape': torch.Size([512, 512]) }, 
        {'peng': self.b4_torgb_affine_bias , 'official': synthesis.b4.torgb.affine.bias , 'shape': torch.Size([512]) }, 
        {'peng': self.b4_torgb_weight , 'official': synthesis.b4.torgb.weight , 'shape': torch.Size([3, 512, 1, 1]) }, 
        {'peng': self.b4_torgb_bias , 'official': synthesis.b4.torgb.bias , 'shape': torch.Size([3]) }, 

        # b8
        {'peng': self.b8_conv0_affine_weight , 'official': synthesis.b8.conv0.affine.weight , 'shape': torch.Size([512, 512]) }, 
        {'peng': self.b8_conv0_affine_bias , 'official': synthesis.b8.conv0.affine.bias , 'shape': torch.Size([512]) }, 
        {'peng': self.b8_conv0_weight , 'official': synthesis.b8.conv0.weight , 'shape': torch.Size([512, 512, 3, 3]) }, 
        # Buffer
        {'peng': self.b8_conv0_noise_const, 'official': synthesis.b8.conv0.noise_const , 'shape': torch.Size([8, 8]) },

        {'peng': self.b8_conv0_noise_strength , 'official': synthesis.b8.conv0.noise_strength , 'shape': torch.Size([ ]) }, 
        {'peng': self.b8_conv0_bias , 'official': synthesis.b8.conv0.bias , 'shape': torch.Size([512]) }, 
        {'peng': self.b8_conv1_affine_weight , 'official': synthesis.b8.conv1.affine.weight , 'shape': torch.Size([512, 512]) }, 
        {'peng': self.b8_conv1_affine_bias , 'official': synthesis.b8.conv1.affine.bias , 'shape': torch.Size([512]) }, 
        {'peng': self.b8_conv1_weight , 'official': synthesis.b8.conv1.weight , 'shape': torch.Size([512, 512, 3, 3]) }, 
        # Buffer
        {'peng': self.b8_conv1_noise_const, 'official': synthesis.b8.conv1.noise_const , 'shape': torch.Size([8, 8]) },

        {'peng': self.b8_conv1_noise_strength , 'official': synthesis.b8.conv1.noise_strength , 'shape': torch.Size([ ]) }, 
        {'peng': self.b8_conv1_bias , 'official': synthesis.b8.conv1.bias , 'shape': torch.Size([512]) }, 
        {'peng': self.b8_torgb_affine_weight , 'official': synthesis.b8.torgb.affine.weight , 'shape': torch.Size([512, 512]) }, 
        {'peng': self.b8_torgb_affine_bias , 'official': synthesis.b8.torgb.affine.bias , 'shape': torch.Size([512]) }, 
        {'peng': self.b8_torgb_weight , 'official': synthesis.b8.torgb.weight , 'shape': torch.Size([3, 512, 1, 1]) }, 
        {'peng': self.b8_torgb_bias , 'official': synthesis.b8.torgb.bias , 'shape': torch.Size([3]) }, 

        # b16
        {'peng': self.b16_conv0_affine_weight , 'official': synthesis.b16.conv0.affine.weight , 'shape': torch.Size([512, 512]) }, 
        {'peng': self.b16_conv0_affine_bias , 'official': synthesis.b16.conv0.affine.bias , 'shape': torch.Size([512]) }, 
        {'peng': self.b16_conv0_weight , 'official': synthesis.b16.conv0.weight , 'shape': torch.Size([512, 512, 3, 3]) }, 
        # Buffer
        {'peng': self.b16_conv0_noise_const, 'official': synthesis.b16.conv0.noise_const , 'shape': torch.Size([16, 16]) },

        {'peng': self.b16_conv0_noise_strength , 'official': synthesis.b16.conv0.noise_strength , 'shape': torch.Size([]) }, 
        {'peng': self.b16_conv0_bias , 'official': synthesis.b16.conv0.bias , 'shape': torch.Size([512]) }, 
        {'peng': self.b16_conv1_affine_weight , 'official': synthesis.b16.conv1.affine.weight , 'shape': torch.Size([512, 512]) }, 
        {'peng': self.b16_conv1_affine_bias , 'official': synthesis.b16.conv1.affine.bias , 'shape': torch.Size([512]) }, 
        {'peng': self.b16_conv1_weight , 'official': synthesis.b16.conv1.weight , 'shape': torch.Size([512, 512, 3, 3]) }, 
        # Buffer
        {'peng': self.b16_conv1_noise_const, 'official': synthesis.b16.conv1.noise_const , 'shape': torch.Size([16, 16]) },

        {'peng': self.b16_conv1_noise_strength , 'official': synthesis.b16.conv1.noise_strength , 'shape': torch.Size([ ]) }, 
        {'peng': self.b16_conv1_bias , 'official': synthesis.b16.conv1.bias , 'shape': torch.Size([512]) }, 
        {'peng': self.b16_torgb_affine_weight , 'official': synthesis.b16.torgb.affine.weight , 'shape': torch.Size([512, 512]) }, 
        {'peng': self.b16_torgb_affine_bias , 'official': synthesis.b16.torgb.affine.bias , 'shape': torch.Size([512]) }, 
        {'peng': self.b16_torgb_weight , 'official': synthesis.b16.torgb.weight , 'shape': torch.Size([3, 512, 1, 1]) }, 
        {'peng': self.b16_torgb_bias , 'official': synthesis.b16.torgb.bias , 'shape': torch.Size([3]) }, 

        # b32
        {'peng': self.b32_conv0_affine_weight , 'official': synthesis.b32.conv0.affine.weight , 'shape': torch.Size([512, 512]) }, 
        {'peng': self.b32_conv0_affine_bias , 'official': synthesis.b32.conv0.affine.bias , 'shape': torch.Size([512]) }, 
        {'peng': self.b32_conv0_weight , 'official': synthesis.b32.conv0.weight , 'shape': torch.Size([512, 512, 3, 3]) }, 
        # Buffer
        {'peng': self.b32_conv0_noise_const, 'official': synthesis.b32.conv0.noise_const , 'shape': torch.Size([32, 32]) },

        {'peng': self.b32_conv0_noise_strength , 'official': synthesis.b32.conv0.noise_strength , 'shape': torch.Size([ ]) }, 
        {'peng': self.b32_conv0_bias , 'official': synthesis.b32.conv0.bias , 'shape': torch.Size([512]) }, 
        {'peng': self.b32_conv1_affine_weight , 'official': synthesis.b32.conv1.affine.weight , 'shape': torch.Size([512, 512]) }, 
        {'peng': self.b32_conv1_affine_bias , 'official': synthesis.b32.conv1.affine.bias , 'shape': torch.Size([512]) }, 
        {'peng': self.b32_conv1_weight , 'official': synthesis.b32.conv1.weight , 'shape': torch.Size([512, 512, 3, 3]) }, 
        # Buffer
        {'peng': self.b32_conv1_noise_const, 'official': synthesis.b32.conv1.noise_const , 'shape': torch.Size([32, 32]) },

        {'peng': self.b32_conv1_noise_strength , 'official': synthesis.b32.conv1.noise_strength , 'shape': torch.Size([ ]) }, 
        {'peng': self.b32_conv1_bias , 'official': synthesis.b32.conv1.bias , 'shape': torch.Size([512]) }, 
        {'peng': self.b32_torgb_affine_weight , 'official': synthesis.b32.torgb.affine.weight , 'shape': torch.Size([512, 512]) }, 
        {'peng': self.b32_torgb_affine_bias , 'official': synthesis.b32.torgb.affine.bias , 'shape': torch.Size([512]) }, 
        {'peng': self.b32_torgb_weight , 'official': synthesis.b32.torgb.weight , 'shape': torch.Size([3, 512, 1, 1]) }, 
        {'peng': self.b32_torgb_bias , 'official': synthesis.b32.torgb.bias , 'shape': torch.Size([3]) }, 

        # b64
        {'peng': self.b64_conv0_affine_weight , 'official': synthesis.b64.conv0.affine.weight , 'shape': torch.Size([512, 512]) }, 
        {'peng': self.b64_conv0_affine_bias , 'official': synthesis.b64.conv0.affine.bias , 'shape': torch.Size([512]) }, 
        {'peng': self.b64_conv0_weight , 'official': synthesis.b64.conv0.weight , 'shape': torch.Size([512, 512, 3, 3]) }, 
        # Buffer
        {'peng': self.b64_conv0_noise_const, 'official': synthesis.b64.conv0.noise_const , 'shape': torch.Size([64, 64]) },

        {'peng': self.b64_conv0_noise_strength , 'official': synthesis.b64.conv0.noise_strength , 'shape': torch.Size([ ]) }, 
        {'peng': self.b64_conv0_bias , 'official': synthesis.b64.conv0.bias , 'shape': torch.Size([512]) }, 
        {'peng': self.b64_conv1_affine_weight , 'official': synthesis.b64.conv1.affine.weight , 'shape': torch.Size([512, 512]) }, 
        {'peng': self.b64_conv1_affine_bias , 'official': synthesis.b64.conv1.affine.bias , 'shape': torch.Size([512]) }, 
        {'peng': self.b64_conv1_weight , 'official': synthesis.b64.conv1.weight , 'shape': torch.Size([512, 512, 3, 3]) }, 
        # Buffer
        {'peng': self.b64_conv1_noise_const, 'official': synthesis.b64.conv1.noise_const , 'shape': torch.Size([64, 64]) },

        {'peng': self.b64_conv1_noise_strength , 'official': synthesis.b64.conv1.noise_strength , 'shape': torch.Size([ ]) }, 
        {'peng': self.b64_conv1_bias , 'official': synthesis.b64.conv1.bias , 'shape': torch.Size([512]) }, 
        {'peng': self.b64_torgb_affine_weight , 'official': synthesis.b64.torgb.affine.weight , 'shape': torch.Size([512, 512]) }, 
        {'peng': self.b64_torgb_affine_bias , 'official': synthesis.b64.torgb.affine.bias , 'shape': torch.Size([512]) }, 
        {'peng': self.b64_torgb_weight , 'official': synthesis.b64.torgb.weight , 'shape': torch.Size([3, 512, 1, 1]) }, 
        {'peng': self.b64_torgb_bias , 'official': synthesis.b64.torgb.bias , 'shape': torch.Size([3]) }, 

        # b128
        {'peng': self.b128_conv0_affine_weight , 'official': synthesis.b128.conv0.affine.weight , 'shape': torch.Size([512, 512]) }, 
        {'peng': self.b128_conv0_affine_bias , 'official': synthesis.b128.conv0.affine.bias , 'shape': torch.Size([512]) }, 
        {'peng': self.b128_conv0_weight , 'official': synthesis.b128.conv0.weight , 'shape': torch.Size([256, 512, 3, 3]) }, 
        # Buffer
        {'peng': self.b128_conv0_noise_const, 'official': synthesis.b128.conv0.noise_const , 'shape': torch.Size([128, 128]) },

        {'peng': self.b128_conv0_noise_strength , 'official': synthesis.b128.conv0.noise_strength , 'shape': torch.Size([ ]) }, 
        {'peng': self.b128_conv0_bias , 'official': synthesis.b128.conv0.bias , 'shape': torch.Size([256]) }, 
        {'peng': self.b128_conv1_affine_weight , 'official': synthesis.b128.conv1.affine.weight , 'shape': torch.Size([256, 512]) }, 
        {'peng': self.b128_conv1_affine_bias , 'official': synthesis.b128.conv1.affine.bias , 'shape': torch.Size([256]) }, 
        {'peng': self.b128_conv1_weight , 'official': synthesis.b128.conv1.weight , 'shape': torch.Size([256, 256, 3, 3]) }, 
        # Buffer
        {'peng': self.b128_conv1_noise_const, 'official': synthesis.b128.conv1.noise_const , 'shape': torch.Size([128, 128]) },

        {'peng': self.b128_conv1_noise_strength , 'official': synthesis.b128.conv1.noise_strength , 'shape': torch.Size([ ]) }, 
        {'peng': self.b128_conv1_bias , 'official': synthesis.b128.conv1.bias , 'shape': torch.Size([256]) }, 
        {'peng': self.b128_torgb_affine_weight , 'official': synthesis.b128.torgb.affine.weight , 'shape': torch.Size([256, 512]) }, 
        {'peng': self.b128_torgb_affine_bias , 'official': synthesis.b128.torgb.affine.bias , 'shape': torch.Size([256]) }, 
        {'peng': self.b128_torgb_weight , 'official': synthesis.b128.torgb.weight , 'shape': torch.Size([3, 256, 1, 1]) }, 
        {'peng': self.b128_torgb_bias , 'official': synthesis.b128.torgb.bias , 'shape': torch.Size([3]) }, 

        # 256
        {'peng': self.b256_conv0_affine_weight , 'official': synthesis.b256.conv0.affine.weight , 'shape': torch.Size([256, 512]) }, 
        {'peng': self.b256_conv0_affine_bias , 'official': synthesis.b256.conv0.affine.bias , 'shape': torch.Size([256]) }, 
        {'peng': self.b256_conv0_weight , 'official': synthesis.b256.conv0.weight , 'shape': torch.Size([128, 256, 3, 3]) }, 
        # Buffer
        {'peng': self.b256_conv0_noise_const, 'official': synthesis.b256.conv0.noise_const , 'shape': torch.Size([256, 256]) },

        {'peng': self.b256_conv0_noise_strength , 'official': synthesis.b256.conv0.noise_strength , 'shape': torch.Size([ ]) }, 
        {'peng': self.b256_conv0_bias , 'official': synthesis.b256.conv0.bias , 'shape': torch.Size([128]) }, 
        {'peng': self.b256_conv1_affine_weight , 'official': synthesis.b256.conv1.affine.weight , 'shape': torch.Size([128, 512]) }, 
        {'peng': self.b256_conv1_affine_bias , 'official': synthesis.b256.conv1.affine.bias , 'shape': torch.Size([128]) }, 
        {'peng': self.b256_conv1_weight , 'official': synthesis.b256.conv1.weight , 'shape': torch.Size([128, 128, 3, 3]) }, 
        # Buffer
        {'peng': self.b256_conv1_noise_const, 'official': synthesis.b256.conv1.noise_const , 'shape': torch.Size([256, 256]) },

        {'peng': self.b256_conv1_noise_strength , 'official': synthesis.b256.conv1.noise_strength , 'shape': torch.Size([ ]) }, 
        {'peng': self.b256_conv1_bias , 'official': synthesis.b256.conv1.bias , 'shape': torch.Size([128]) }, 
        {'peng': self.b256_torgb_affine_weight , 'official': synthesis.b256.torgb.affine.weight , 'shape': torch.Size([128, 512]) }, 
        {'peng': self.b256_torgb_affine_bias , 'official': synthesis.b256.torgb.affine.bias , 'shape': torch.Size([128]) }, 
        {'peng': self.b256_torgb_weight , 'official': synthesis.b256.torgb.weight , 'shape': torch.Size([3, 128, 1, 1]) }, 
        {'peng': self.b256_torgb_bias , 'official': synthesis.b256.torgb.bias , 'shape': torch.Size([3]) }, 

        # 512
        {'peng': self.b512_conv0_affine_weight , 'official': synthesis.b512.conv0.affine.weight , 'shape': torch.Size([128, 512]) }, 
        {'peng': self.b512_conv0_affine_bias , 'official': synthesis.b512.conv0.affine.bias , 'shape': torch.Size([128]) }, 
        {'peng': self.b512_conv0_weight , 'official': synthesis.b512.conv0.weight , 'shape': torch.Size([64, 128, 3, 3]) }, 
        # Buffer
        {'peng': self.b512_conv0_noise_const, 'official': synthesis.b512.conv0.noise_const , 'shape': torch.Size([512, 512]) },

        {'peng': self.b512_conv0_noise_strength , 'official': synthesis.b512.conv0.noise_strength , 'shape': torch.Size([]) }, 
        {'peng': self.b512_conv0_bias , 'official': synthesis.b512.conv0.bias , 'shape': torch.Size([64]) }, 
        {'peng': self.b512_conv1_affine_weight , 'official': synthesis.b512.conv1.affine.weight , 'shape': torch.Size([64, 512]) }, 
        {'peng': self.b512_conv1_affine_bias , 'official': synthesis.b512.conv1.affine.bias , 'shape': torch.Size([64]) }, 
        {'peng': self.b512_conv1_weight , 'official': synthesis.b512.conv1.weight , 'shape': torch.Size([64, 64, 3, 3]) }, 
        # Buffer
        {'peng': self.b512_conv1_noise_const, 'official': synthesis.b512.conv1.noise_const , 'shape': torch.Size([512, 512]) },

        {'peng': self.b512_conv1_noise_strength , 'official': synthesis.b512.conv1.noise_strength , 'shape': torch.Size([ ]) }, 
        {'peng': self.b512_conv1_bias , 'official': synthesis.b512.conv1.bias , 'shape': torch.Size([64]) }, 
        {'peng': self.b512_torgb_affine_weight , 'official': synthesis.b512.torgb.affine.weight , 'shape': torch.Size([64, 512]) }, 
        {'peng': self.b512_torgb_affine_bias , 'official': synthesis.b512.torgb.affine.bias , 'shape': torch.Size([64]) }, 
        {'peng': self.b512_torgb_weight , 'official': synthesis.b512.torgb.weight , 'shape': torch.Size([3, 64, 1, 1]) }, 
        {'peng': self.b512_torgb_bias , 'official': synthesis.b512.torgb.bias , 'shape': torch.Size([3]) }, 

        # b1024
        {'peng': self.b1024_conv0_affine_weight , 'official': synthesis.b1024.conv0.affine.weight , 'shape': torch.Size([64, 512]) }, 
        {'peng': self.b1024_conv0_affine_bias , 'official': synthesis.b1024.conv0.affine.bias , 'shape': torch.Size([64]) }, 
        {'peng': self.b1024_conv0_weight , 'official': synthesis.b1024.conv0.weight , 'shape': torch.Size([32, 64, 3, 3]) }, 
        # Buffer
        {'peng': self.b1024_conv0_noise_const, 'official': synthesis.b1024.conv0.noise_const , 'shape': torch.Size([1024, 1024]) },

        {'peng': self.b1024_conv0_noise_strength , 'official': synthesis.b1024.conv0.noise_strength , 'shape': torch.Size([ ]) }, 
        {'peng': self.b1024_conv0_bias , 'official': synthesis.b1024.conv0.bias , 'shape': torch.Size([32]) }, 
        {'peng': self.b1024_conv1_affine_weight , 'official': synthesis.b1024.conv1.affine.weight , 'shape': torch.Size([32, 512]) }, 
        {'peng': self.b1024_conv1_affine_bias , 'official': synthesis.b1024.conv1.affine.bias , 'shape': torch.Size([32]) }, 
        {'peng': self.b1024_conv1_weight , 'official': synthesis.b1024.conv1.weight , 'shape': torch.Size([32, 32, 3, 3]) }, 
        # Buffer
        {'peng': self.b1024_conv1_noise_const, 'official': synthesis.b1024.conv1.noise_const , 'shape': torch.Size([1024, 1024]) },

        {'peng': self.b1024_conv1_noise_strength , 'official': synthesis.b1024.conv1.noise_strength , 'shape': torch.Size([ ]) }, 
        {'peng': self.b1024_conv1_bias , 'official': synthesis.b1024.conv1.bias , 'shape': torch.Size([32]) }, 
        {'peng': self.b1024_torgb_affine_weight , 'official': synthesis.b1024.torgb.affine.weight , 'shape': torch.Size([32, 512]) }, 
        {'peng': self.b1024_torgb_affine_bias , 'official': synthesis.b1024.torgb.affine.bias , 'shape': torch.Size([32]) }, 
        {'peng': self.b1024_torgb_weight , 'official': synthesis.b1024.torgb.weight , 'shape': torch.Size([3, 32, 1, 1]) }, 
        {'peng': self.b1024_torgb_bias , 'official': synthesis.b1024.torgb.bias , 'shape': torch.Size([3]) }, 

        ]
        for pair in mapping_dict:
            tensor=pair['peng']
            src_tensors=pair['official']
            assert src_tensors.shape==pair['shape']
            tensor.copy_(src_tensors.detach()).requires_grad_(tensor.requires_grad)
        return 
    
layer_IO_dict={
    'b4':{'input_x':None,'input_img':None,'output_x':[1, 512, 4, 4],'output_img':[1, 3, 4, 4]},
    'b8':{'input_x':[1, 512, 4, 4],'input_img':[1, 3, 4, 4],'output_x':[1, 512, 8, 8],'output_img':[1, 3, 8, 8]},
    'b16':{'input_x':[1, 512, 8, 8],'input_img':[1, 3, 8, 8],'output_x':[1, 512, 16,16],'output_img':[1, 3, 16, 16]},
    'b32':{'input_x':[1, 512, 16,16],'input_img':[1, 3, 16, 16],'output_x':[1, 512, 32,32],'output_img':[1, 3, 32, 32]},
    'b64':{'input_x':[1, 512, 32,32],'input_img':[1, 3, 32, 32],'output_x':[1, 512, 64,64],'output_img':[1, 3, 64, 64]},
    'b128':{'input_x':[1, 512, 64,64],'input_img':[1, 3, 64, 64],'output_x':[1, 256, 128,128],'output_img':[1, 3, 128, 128]},
    'b256':{'input_x':[1, 256, 128,128],'input_img':[1, 3, 128, 128],'output_x':[1, 128, 256,256],'output_img':[1, 3, 256, 256]},
    'b512':{'input_x':[1, 128, 256,256],'input_img':[1, 3, 256, 256],'output_x':[1, 64, 512,512],'output_img':[1, 3, 512, 512]},
    'b1024':{'input_x':[1, 64, 512,512],'input_img':[1, 3, 512, 512],'output_x':[1, 32, 1024,1024],'output_img':[1, 3, 1024, 1024]},
}

def img_test(Mapping_pkl,G_test,device):
    input=torch.randn(1,512).to(device)
    ws=Mapping_pkl(input)
    img=G_test(ws, noise_mode='const').detach().cpu()
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'test.png')

def test_layer(S1,S2,layer='b1024'):
    device=torch.device('cuda:0')
    ws = torch.randn(1,18,512).to(device)
    input_x= torch.randn(layer_IO_dict[layer]['input_x']).to(device)
    input_img=torch.randn(layer_IO_dict[layer]['input_img']).to(device)
    S1_layer = getattr(S1, layer)
    S2_layer = getattr(S2, layer)
    x1, img1 = S1_layer(input_x, input_img, ws.narrow(1, 1, 3), force_fp32=True, noise_mode='const')
    x2, img2 = S2_layer(input_x, input_img, ws.narrow(1, 1, 3), force_fp32=True, noise_mode='const')
    print(x1.equal(x2))
    # print(x1-x2)
    print(x1.shape,img1.shape)

if __name__ == '__main__':
    device=torch.device('cuda:0')
    ws=torch.randn(1,18,512).to(device)
    network_pkl=r"/data/network-snapshot-000129.pkl"

    G_official=peng_network.SynthesisNetwork(
        w_dim=512, channel_base=32768, channel_max=512, num_fp16_res=4, conv_clamp=256, img_resolution=1024,
        img_channels=3).requires_grad_(False).eval().to(device)
    G_test = SynthesisNet().requires_grad_(False).eval().to(device)
    with dnnlib.util.open_url(network_pkl) as f:
        G=legacy.load_network_pkl(f)['G_ema']
        Synthesis_pkl = G.synthesis
        Mapping_pkl=G.mapping.eval().to(device)
    Synthesis_pkl = Synthesis_pkl.requires_grad_(False).eval().to(device)
    misc.copy_params_and_buffers(Synthesis_pkl, G_official, require_all=False)
    G_test.copy_param_and_buffer(Synthesis_pkl)
    buffer=torch.tensor([
        [0.015625,0.046875,0.046875,0.015625],
        [0.046875,0.140625, 0.140625, 0.046875],
        [0.046875, 0.140625, 0.140625, 0.046875],
        [0.015625, 0.046875, 0.046875, 0.015625]]).to(torch.float32).to(device)
    print(buffer.equal(Synthesis_pkl.b4.resample_filter))
    print(buffer.equal(Synthesis_pkl.b1024.resample_filter))

