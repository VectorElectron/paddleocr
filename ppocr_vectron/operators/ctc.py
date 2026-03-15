import torch, torch.nn as nn
import torch.nn.functional as F

def ctc_decode_flatten(probs, blank_id=0):
    max_probs, max_indices = torch.max(probs, dim=-1)
    prev_indices = torch.cat([
        torch.tensor([-1], dtype=max_indices.dtype, device=max_indices.device), 
        max_indices[:-1]
    ])
    diff_mask = max_indices != prev_indices
    not_blank_mask = max_indices != blank_id
    valid_mask = diff_mask & not_blank_mask
    final_indices = torch.where(
        valid_mask, 
        max_indices, 
        torch.tensor(-1, dtype=max_indices.dtype, device=max_indices.device)
    )
    return final_indices[valid_mask].to(torch.int32), max_probs[valid_mask].mean()


class CTCDecoder(nn.Module):
    def __init__(self, blank_idx=0):
        super().__init__()
        self.blank_idx = blank_idx

    def forward(self, logits):
        return ctc_decode_flatten(logits, self.blank_idx)


if __name__ == '__main__':
    model = CTCDecoder(blank_idx=0)
    model.eval()

    # 模拟输入
    dummy_logits = torch.randn(40, 1000)

    torch.onnx.export(
        model,
        (dummy_logits,),
        "../model/ctc_decode.onnx",
        input_names=['x'],
        output_names=['out_indices', 'out_scores'],
        dynamic_axes={
            'x': {0: 'time_steps', 1:'features'},
            'out_indices': {0: 'total_valid_chars'}, # 动态输出长度
        },
        dynamo=False,
        opset_version=16
    )

