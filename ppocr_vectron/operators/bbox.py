import torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F

def bincount(x, weights=None, minlength=None, mode='sum'):
    # return torch.bincount(x, weights, minlength)
    if minlength is None: minlength = x.max()+1
    dtype = torch.int32 if weights is None else torch.float32
    result = torch.zeros(minlength, dtype=dtype)
    if weights is None: weights = torch.ones_like(x, dtype=torch.int32)
    return result.scatter_reduce(0, x.to(torch.long), weights, reduce=mode)

def label(msk):
    H, W = msk.shape
    
    num_pixels = msk.sum().to(torch.long)
    lab = torch.zeros((H, W), dtype=torch.long)
    init_vals = torch.arange(1, num_pixels + 1, dtype=torch.long)
    lab[msk] = init_vals
    
    lut = torch.arange(num_pixels + 1, dtype=torch.long)

    steps = [12, 6, 3]
    for s in steps:
        lab_tensor = lab.view(1, 1, H, W).float()
        maxlab_tensor = F.max_pool2d(
            lab_tensor, kernel_size=3, stride=1, padding=1)
        maxlab = maxlab_tensor.view(H, W).long()
        update_mask = maxlab > lab
        
        # 更新 LUT
        indices = lab[update_mask]
        lut[indices] = maxlab[update_mask]
        lut[0] = 0
        
        # 路径压缩
        for _ in range(s): lut = lut[lut]
        lab = lut[lab]

    # 3. 归一化
    _, indices = torch.unique(lab, return_inverse=True)
    return indices.reshape(H, W).to(torch.int32)

def svd_2d(cov_flat):
    var_x = cov_flat[:, 0]      # x的方差
    cov_xy = cov_flat[:, 1]     # x和y的协方差
    var_y = cov_flat[:, 3]      # y的方差
    
    numerator = 2.0 * cov_xy
    denominator = var_x - var_y
    
    # 使用atan2避免除零问题
    theta = 0.5 * torch.atan2(numerator, denominator)
    
    # 2. 计算旋转矩阵
    # R = [[cosθ, -sinθ], [sinθ, cosθ]]
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    
    # 构造旋转矩阵 (n, 2, 2)
    n = cov_flat.shape[0]
    rotation = torch.zeros((n, 2, 2), device=cov_flat.device)
    
    rotation[:, 0, 0] = cos_theta      # R[0,0] = cosθ
    rotation[:, 0, 1] = -sin_theta     # R[0,1] = -sinθ
    rotation[:, 1, 0] = sin_theta      # R[1,0] = sinθ
    rotation[:, 1, 1] = cos_theta      # R[1,1] = cosθ
    
    trace = var_x + var_y
    discriminant = (var_x - var_y)**2 + 4.0 * cov_xy**2
    
    # 确保判别式非负（理论上对称矩阵应该非负，但数值计算可能有问题）
    discriminant = torch.max(discriminant, torch.tensor(1e-12))
    sqrt_disc = torch.sqrt(discriminant)
    
    # 计算特征值（方差）
    lambda1 = (trace + sqrt_disc) / 2.0  # 较大特征值，长轴方差
    lambda2 = (trace - sqrt_disc) / 2.0  # 较小特征值，短轴方差
    
    # 确保特征值非负（避免数值误差）
    lambda1 = torch.max(lambda1, torch.tensor(0.0))
    lambda2 = torch.max(lambda2, torch.tensor(0.0))
    
    # 计算轴长度（标准差 = 方差的平方根）
    axis1_length = torch.sqrt(lambda1)  # 长轴长度
    axis2_length = torch.sqrt(lambda2)  # 短轴长度
    
    axis_lengths = torch.stack([axis1_length, axis2_length], dim=-1)
    return rotation, axis_lengths, theta

def bbox(hot, thr=0.3, boxthr=0.7, sizethr=5, mar=2, maxn=100):
    lab = label(hot>thr)
    lab[lab>=maxn] = 0
    n_regions = maxn.to(torch.int64)
    ys, xs = torch.meshgrid(
        torch.arange(hot.shape[0]), torch.arange(hot.shape[1]), indexing='ij')
    
    lab_flat = lab.flatten()
    x_flat = xs.flatten().to(torch.float32)
    y_flat = ys.flatten().to(torch.float32)

    # 计算每个区域的像素数
    area = bincount(lab_flat, minlength=n_regions).to(torch.float32)

    # 计算均值
    mu_x = bincount(lab_flat, x_flat, minlength=n_regions) / area
    mu_y = bincount(lab_flat, y_flat, minlength=n_regions) / area
    mu_xy = torch.stack([mu_x, mu_y], dim=1)

    center_x = x_flat - mu_x[lab_flat]
    center_y = y_flat - mu_y[lab_flat]

    # 计算二阶矩
    xx = bincount(lab_flat, center_x * center_x, minlength=n_regions) / area
    xy = bincount(lab_flat, center_x * center_y, minlength=n_regions) / area
    yy = bincount(lab_flat, center_y * center_y, minlength=n_regions) / area

    # 构造协方差矩阵
    cov = torch.stack([xx, xy, xy, yy]).T
    m, l, _ = svd_2d(cov)
    margin = l[:,1] ** 0.5 * 4 * mar

    # 每个点中心化，并旋转到正交位置
    coords = torch.stack([x_flat, y_flat], dim=1)  # (n_pixels, 2)
    coords = coords - mu_xy[lab_flat]
    rotaed_xy = (m[lab_flat].transpose(1, 2) @ coords.unsqueeze(-1)).squeeze(-1)

    # 计算上下左右边界
    minx = bincount(lab_flat, rotaed_xy[:, 0], minlength=n_regions, mode='amin') - margin
    maxx = bincount(lab_flat, rotaed_xy[:, 0], minlength=n_regions, mode='amax') + margin
    miny = bincount(lab_flat, rotaed_xy[:, 1], minlength=n_regions, mode='amin') - margin
    maxy = bincount(lab_flat, rotaed_xy[:, 1], minlength=n_regions, mode='amax') + margin

    # miny -= margin; maxx += margin; maxy += margin

    # 旋转回图像坐标系
    recs = torch.stack([minx, miny, maxx, miny, maxx, maxy, minx, maxy, minx, miny], dim=-1)
    recs = recs.reshape(-1,5,2).transpose(1,2)
    newxy = (m @ recs + mu_xy[:,:,None]).transpose(1,2)

    # 过滤
    level = bincount(lab_flat, hot.flatten(), minlength=n_regions)
    # print(area.dtype)
    msk1 = (level / area) > boxthr
    msk2 = ((maxx - minx) > sizethr) & ((maxy - miny) > sizethr)
    return newxy[msk1 & msk2]


class BBoxModel(nn.Module):
    def forward(self, hot, scale, thr=0.3, boxthr=0.7, sizethr=5.0, mar=1.0, maxn=100):
        return bbox(hot, thr, boxthr, sizethr, mar, maxn) * scale * 2

'''
from imageio import imread
msk = imread('./demo/mask.png')
msk = torch.tensor(msk, dtype=torch.float32)/255
'''

if __name__ == '__main__':
    model = BBoxModel()
    model.eval()

    dummy_input = torch.zeros((256, 256), dtype=torch.float32)
    dummy_input[20:80, 20:40] = 1

    scale = torch.ones(2, dtype=torch.float32)

    thr, boxthr, sizethr, mar = torch.tensor([0.3,0.7,3.0,1.0], dtype=torch.float32)
    maxn = torch.tensor(100, dtype=torch.int32)

    # 导出 ONNX
    torch.onnx.export(
        model,
        (dummy_input, scale, thr, boxthr, sizethr, mar, maxn),
        "../model/bbox_extract.onnx",
        dynamo=False,
        opset_version=18,
        input_names=['hotimg', 'scale', 'thr', 'boxthr', 'sizethr', 'mar', 'maxn'],
        output_names=['boxes'],
        dynamic_axes={'hotimg': {0: 'height', 1: 'width'}, 'boxes': {0: 'height', 1: 'width'}}
    )


