from torch import nn
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
        )
        
class ConvReluUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.make_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, (3, 3), stride=1, padding=1, bias=False
            ),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.block(x)
        if self.upsample:
            x = self.make_upsample(x)
        return x



class SegmentationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_upsamples=0):
        super().__init__()

        blocks = [ConvReluUpsample(in_channels, out_channels, upsample=bool(n_upsamples))]

        if n_upsamples > 1:
            for _ in range(1, n_upsamples):
                blocks.append(ConvReluUpsample(out_channels, out_channels, upsample=True))

        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.block(x)
        
class FPN(nn.Module):        
    def __init__(self, n_classes=1, 
                 pyramid_channels=256, 
                 segmentation_channels=256,
                 conv_down_init = 64,conv_down_count = 4):
        super().__init__()
        n_smooth = conv_down_count - 1
        n_lateral = conv_down_count - 1
        upsamples = list(range(n_lateral))
        # Bottom-up layers
        
        down_in_size, down_out_size = self.create_bottom_up(conv_down_init,conv_down_count)
        lateral_init = down_in_size
        self.maxpool = nn.MaxPool2d(2)
        
        # Top layer
        self.toplayer = nn.Conv2d(down_out_size, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.create_smooth(n_smooth)

        # Lateral layers
        self.create_lateral(n_lateral,lateral_init)

        # Segmentation block layers
        self.seg_blocks = nn.ModuleList([
            SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=n_upsamples)
            for n_upsamples in upsamples
        ])
        
        # Last layer
        self.last_conv = nn.Conv2d(256, n_classes, kernel_size=1, stride=1, padding=0)
        def create_bottom_up(self,conv_down_init = 64,conv_down_count = 4):
        self.conv_down_count = conv_down_count

        self.conv_down1 = double_conv(3, conv_down_init)
        for i in range(conv_down_count):
          down_in_size = conv_down_init* (2**i)
          down_out_size = conv_down_init* (2**(i+1))
          print(f'create_bottom_up:: down_in_size: {down_in_size}, down_out_size: {down_out_size}')
          setattr(self,f'conv_down{i+2}',double_conv(down_in_size, down_out_size))
        return down_in_size, down_out_size

    def create_smooth(self,n_smooth = 3):
        self.n_smooth = n_smooth
        for i_s in range(1,n_smooth+1):
          print(f'create_smooth_layers:: smooth{i_s} ')
          setattr(self,f'smooth{i_s}',nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
          
    def create_lateral(self,n_lateral = 3,lateral_init = 512):
        self.n_lateral = n_lateral
        for i_l in range(n_lateral):
          lateral_cur = int(lateral_init*(1/(2**i_l)))
          print(f'create_lateral:: lateral_cur: {lateral_cur}')
          setattr(self,f'latlayer{i_l+1}',nn.Conv2d(lateral_cur, 256, kernel_size=1, stride=1, padding=0))

    def upsample_add(self, x, y):
        _,_,H,W = y.size()
        upsample = nn.Upsample(size=(H,W), mode='bilinear', align_corners=True) 
        
        return upsample(x) + y
    
    def upsample(self, x, h, w):
        sample = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)
        return sample(x)

    def forward_conv_down(self,x):
        """
        1,2,3,4,5
        """
        outputs = []
        c1 = self.maxpool(self.conv_down1(x))
        outputs.append({'c1':c1})
        for i in range(self.conv_down_count):
          conv_down_name = f'conv_down{i+2}'
          print(f'forward_conv_down:: getting {conv_down_name}')
          current_layer = getattr(self,conv_down_name)
          c = self.maxpool(current_layer(outputs[-1][f'c{i+1}']))
          outputs.append({f'c{i+2}':c})
        print('forward_conv_down:: {}'.format([x.keys() for x in outputs]))
        return outputs

    def forward_top_down(self,conv_down_outputs):
        """
        5,4,3,2
        """
        p_top = self.toplayer(conv_down_outputs[-1][f'c{len(conv_down_outputs)}'])
        top_down_outputs = []
        top_down_outputs.append({f'p{len(conv_down_outputs)}':p_top})

        for i in range(1,len(conv_down_outputs)-1):
          ii = len(conv_down_outputs)-1-i
          print(f'forward_top_down:: at down:{ii}, lateral: {i}')
          
          conv_down_current = conv_down_outputs[ii][f'c{ii+1}']
          
          current_lateral = getattr(self,f'latlayer{i}')
          
          prev_top_down_output = top_down_outputs[-1][f'p{ii+2}']
          p = self.upsample_add(prev_top_down_output, current_lateral(conv_down_current))   
          top_down_outputs.append({f'p{ii+1}':p})
        print('forward_top_down:: {}'.format([x.keys() for x in top_down_outputs]))
        return top_down_outputs

    def forward_smooth(self,top_down_outputs):
        for i in range(1,len(top_down_outputs)):
          ii = len(top_down_outputs)-i
          print(f'forward_smooth:: in p{ii+1}, smooth{i}')
          current_smooth_layer = getattr(self,f'smooth{i}')
          top_down_outputs[i][f'p{ii+1}'] = current_smooth_layer(top_down_outputs[i][f'p{ii+1}'])
        return top_down_outputs

    def get_final_p_outputs(self,top_down_outputs_smooth):
        final_p = []
        for i in range(len(top_down_outputs_smooth)):
          ii = len(top_down_outputs_smooth) - i -1 
          current_p = top_down_outputs_smooth[ii]
          print(f'get_final_p_outputs:: {current_p.keys()}')
          final_p.append(current_p[f'p{i+2}'])
        return final_p

    def forward(self, x):
        
        # Bottom-up
        conv_down_outputs = self.forward_conv_down(x)

        # Top-down
        top_down_outputs = self.forward_top_down(conv_down_outputs)

        # Smooth
        top_down_outputs_smooth = self.forward_smooth(top_down_outputs)
        
        # Segmentation
        _, _, h, w = top_down_outputs_smooth[-1]['p2'].size()
        final_p = self.get_final_p_outputs(top_down_outputs_smooth)
        feature_pyramid = [seg_block(p) for seg_block, p in zip(self.seg_blocks, final_p)]
        
        final_p_count = len(final_p)
        out = self.upsample(self.last_conv(sum(feature_pyramid)), final_p_count * h, final_p_count * w)
        
        out = torch.sigmoid(out)
        return out
