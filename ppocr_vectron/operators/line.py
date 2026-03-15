import torch, torch.nn as nn
import torch.nn.functional as F

def extracts(img, boxes, scale, height=32):
    device = img.device
    boxes = boxes / scale
    num_boxes = boxes.shape[0]
    img_c, img_h, img_w = img.shape  # <-- 修改点：解包 C, H, W

    # 1. 计算每个 box 的目标宽度
    h_dists = torch.norm(boxes[:, 2] - boxes[:, 1], dim=1)
    w_dists = torch.norm(boxes[:, 1] - boxes[:, 0], dim=1)
    widths = (height * w_dists / h_dists.clamp(min=1e-6)).long()
    max_w = widths.max()//8*8+8

    # 2. 生成归一化网格基础
    rows = torch.arange(height, device=device).float() / (height - 1)
    cols = torch.arange(max_w, device=device).float()
    y, x_idx = torch.meshgrid(rows, cols, indexing='ij')
    
    x_idx = x_idx.unsqueeze(0).expand(num_boxes, -1, -1)
    y = y.unsqueeze(0).expand(num_boxes, -1, -1)
    
    curr_w = widths.view(-1, 1, 1).float()
    x = x_idx / (curr_w - 1.0).clamp(min=1.0)

    # 3. 严格遵循原著的双线性插值逻辑
    p0, p1, p2, p3 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    
    # 辅助内部插值函数，保持代码整洁
    def _interp(idx):
        v0, v1, v2, v3 = p0[:, idx, None, None], p1[:, idx, None, None], \
                         p2[:, idx, None, None], p3[:, idx, None, None]
        return (1-x)*(1-y)*v0 + x*(1-y)*v1 + x*y*v2 + (1-x)*y*v3

    res_x = _interp(0)
    res_y = _interp(1)

    # 4. 制作 Mask 并处理越界坐标实现自动补 0
    mask = x_idx < curr_w
    grid_c = (res_x / (img_w - 1.0)) * 2.0 - 1.0
    grid_r = (res_y / (img_h - 1.0)) * 2.0 - 1.0
    
    grid_c = torch.where(mask, grid_c, torch.tensor(2.0, device=device))
    grid_r = torch.where(mask, grid_r, torch.tensor(2.0, device=device))
    grid = torch.stack([grid_c, grid_r], dim=-1)

    img_4d = img.unsqueeze(0).expand(num_boxes, -1, -1, -1)
    
    output = F.grid_sample(img_4d, grid, mode='nearest', padding_mode='zeros', align_corners=True)
    
    return output, (widths//8*8+8).to(torch.int32)

class Extract(nn.Module):
    def __init__(self, target_height=32):
        super().__init__()
        self.target_height = target_height

    def forward(self, img, boxes, scale):
        return extracts(img, boxes, scale, self.target_height)

if __name__ == '__main__':
    model = Extract(target_height=32)
    model.eval()

    dummy_img = torch.randn(3, 512, 512)
    dummy_boxes = torch.tensor([
        [[10, 10], [100, 10], [100, 40], [10, 40], [10, 10]],
    ], dtype=torch.float32)
    scale = torch.ones(2, dtype=torch.float32)

    # 3. 导出配置
    torch.onnx.export(
        model,
        (dummy_img, dummy_boxes, scale),
        '../model/line_extract.onnx',
        input_names=['x', 'boxes', 'scale'],
        output_names=['blocks', 'widths'],
        dynamo=False,
        dynamic_axes={
            'x': {1: 'img_h', 2: 'img_w'},
            'boxes': {0: 'num_boxes'},
            'out_crops': {0: 'num_boxes', 2: 'max_w'},
            'out_widths': {0: 'num_boxes'}
        },
        opset_version=18
    )

