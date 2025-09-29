import torch
import torch.nn as nn
from torchvision import models, transforms
from apply_lut import *
from palette import *

class WeightPredictor(nn.Module):
    def __init__(self, output_size, mask_num):
        super(WeightPredictor, self).__init__()

        self.output_size = output_size
        self.mask_num = mask_num
        
        self.base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        old_conv = self.base_model.features[0][0]
        
        new_conv = nn.Conv2d(
            in_channels=3+mask_num,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )
        
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = old_conv.weight  # copy RGB weights
            nn.init.kaiming_normal_(new_conv.weight[:, 3:, :, :], mode='fan_out', nonlinearity='relu')  # init for mask channels
        
        self.base_model.features[0][0] = new_conv

        
        self.base_model.classifier[1] = nn.Linear(in_features=1280, out_features=output_size * mask_num)
        # self.softmax = nn.Sigmoid()
        self.softmax = nn.Tanh()
        # self.softmax = nn.Softmax(dim=2)
        
    def forward(self, x):
        pred = self.base_model(x)
        pred = pred.reshape(-1, self.mask_num, self.output_size)
        return self.softmax(pred) * 1.5
    

class ImageEnhancer2(nn.Module):
    def __init__(self, luts, output_size, mask_num):
        super(ImageEnhancer2, self).__init__()

        self.output_size = output_size
        self.mask_num = mask_num
        
        self.weight_predictor = WeightPredictor(output_size, mask_num+1)
        self.mask_generator = PaletteGenerate(k=mask_num)

        self.LUTs = luts
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.LUTs = nn.Parameter(luts)

    def forward(self, x, masks=None):
        # self.LUTs.data.clamp_(0, 1)

        if masks == None:
            masks = []
            for i in range(x.shape[0]):
                mask = self.mask_generator.forward(x[i].unsqueeze(0))[0]
                masks.append(mask)
    
            masks = torch.cat(masks, dim=0)

        base_image_mask = torch.ones(x.shape[0], 1, x.shape[2], x.shape[3])
        base_image_mask = base_image_mask.to(self.device)

        masks = torch.cat((base_image_mask, masks), dim=1)
        concated = torch.cat((x, masks), dim=1)
        concated = F.interpolate(concated, size=(224, 224), mode='bilinear', align_corners=False)
        
        pred = self.weight_predictor(concated)
        
        transformed_output = apply_weighted_luts_to_batch(x, self.LUTs, pred, 33, masks)
    
        return {
            "enhanced_image": transformed_output,
            "pred": pred,
            "masks": masks
        }
