from torch import nn
from .blocks import double_conv
import torch

class UNet(nn.Module):
    def __init__(self, n_classes,n_layers=4,input_channels=3):
        super().__init__()
        self.n_layers = n_layers
        self.input_channels = input_channels
        self.create_down()

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.create_up()

        self.last_conv = nn.Conv2d(64, n_classes, kernel_size=1)
    
    def forward(self, x):
        # Batch - 1d tensor.  N_channels - 1d tensor, IMG_SIZE - 2d tensor.
        # Example: x.shape >>> (10, 3, 256, 256).
        
        x, conv_outputs = self.forward_down(x)
        print(f'conv_outputs count: {len(conv_outputs)}')
        x = self.upsample(x)        # <- BATCH, 512, IMG_SIZE -> BATCH, 512, IMG_SIZE 2x up.
        
        #(Below the same)                                 N this       ==        N this.  Because the first N is upsampled.
        x = torch.cat([x, conv_outputs[-1]], dim=1) # <- BATCH, 512, IMG_SIZE & BATCH, 256, IMG_SIZE--> BATCH, 768, IMG_SIZE.
        
        x = self.forward_up(x,conv_outputs)

        x = self.conv_up1(x) # <- BATCH, 128, IMG_SIZE --> BATCH, 64, IMG_SIZE.
        
        out = self.last_conv(x) # <- BATCH, 64, IMG_SIZE --> BATCH, n_classes, IMG_SIZE.
        out = torch.sigmoid(out)
        
        return out
        
    def create_down(self):
        for i in range(self.n_layers):
          input = self.input_channels if i==0 else int(2**(5+i))
          output = int(2**(6+i))
          print(f'{input},{output}')
          setattr(self,f'conv_down{i+1}',double_conv(input, output))

    def create_up(self):
        for i in range(self.n_layers-1):
          input_init = int(2**(7+i))
          output = int(2**(6+i))
          input_fin = input_init + output
          print(f'{input_init} + {output},{output}')
          setattr(self,f'conv_up{i+1}',double_conv(input_fin, output))
        
    def forward_down(self,x):
        conv_outputs = []
        for i in range(self.n_layers):
          layer_name = f'conv_down{i+1}'
          print(layer_name)
          layer = getattr(self,layer_name)

          if(i<self.n_layers-1):
            conv = layer(x)  # <- BATCH, 3, IMG_SIZE  -> BATCH, 64, IMG_SIZE..
            x = self.maxpool(conv)
            conv_outputs.append(conv)
          else:
            x = layer(x)
        return x, conv_outputs
          
    def forward_up(self,x,conv_outputs):
        less_1_n_layers = self.n_layers-1 
        for i in range(less_1_n_layers):
          ii = less_1_n_layers - i
          if(ii==1):
            break
          layer_name = f'conv_up{ii}'
          print(layer_name)
          layer = getattr(self,layer_name)
          x = layer(x)
          x = self.upsample(x)    
          x = torch.cat([x, conv_outputs[ii-2]], dim=1)  
        return x
