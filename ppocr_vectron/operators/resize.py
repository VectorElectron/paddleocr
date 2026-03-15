import torch
import torch.nn as nn
import torch.nn.functional as F

class ResizeTransDiv(nn.Module):
    def __init__(self, mode='bilinear'):
        super(ResizeTransDiv, self).__init__()
        self.mode = mode
    
    def forward(self, x, dial=640*1.414):
        l = torch.max((x.shape[0]**2 + x.shape[1]**2)**0.5, dial)
        new_h = (x.shape[0]*dial/l/64+0.5).to(torch.long)*64
        new_w = (x.shape[1]*dial/l/64+0.5).to(torch.long)*64
        kx, ky = x.shape[1]/new_w, x.shape[0]/new_h
        x = x.permute(2, 0, 1).unsqueeze(0)
        x = x.to(torch.float32)/255
        resized = F.interpolate(x, size=(new_h, new_w), 
            mode=self.mode, align_corners=False)
        half = F.interpolate(x, size=(new_h//2, new_w//2), 
            mode=self.mode, align_corners=False)
        return resized, half, torch.stack([kx, ky])

if __name__ == '__main__':
    resize_model = ResizeTransDiv(mode='bilinear')

    dummy_input = torch.zeros((256, 256, 3), dtype=torch.uint8)
    dial = torch.tensor(1024, dtype=torch.int32)
    k = torch.tensor(255, dtype=torch.float32)
    torch.onnx.export(
        resize_model,
        (dummy_input, dial),
        "../model/resize_trans.onnx",
        input_names=['image', 'dial'],
        output_names=['output', 'half', 'scale'],
        dynamic_axes={
            'image': {0: 'height', 1: 'width', 2:'channel'},
        },
        opset_version=18,
        dynamo = False,
    )
