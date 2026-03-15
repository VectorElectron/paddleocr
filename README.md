# PP-OCRv5-Pure-ONNX: 全流程算子化端到端 OCR 方案

## 1. 核心贡献：解耦 Paddle 对 Python 环境的强依赖
PaddleOCR 官方在非 Python 环境下（如 C++、WebAssembly）缺乏完整的端到端等效实现。本项目通过将所有后处理几何逻辑重构为 **ONNX 静态图算子**，填补了这一空白。

### 核心技术点
* **算子化连通域 (Labeling)**：在 `bbox.py` 中利用 Tensor 形态学逻辑实现像素级标记，完全脱离 OpenCV。
* **SVD 区域提取**：通过对像素坐标分布进行 **奇异值分解 (SVD)** 提取主成分方向，取代传统的边缘拟合。SVD 方案在不同硬件平台上具有完美的计算一致性。
* **Grid Sample 行提取**：在 `line.py` 中利用 **网格采样 (Grid Sample)** 实现亚像素级的仿射变换，一次性完成旋转校正与裁剪。
* **全算子 CTC 解码**：在 `ctc.py` 中将 Greedy Search 封装进模型，输出即为文字索引。

## 2. Python 完整闭环 Demo
```python
import onnxruntime as ort
import numpy as np
from PIL import Image

# 1. 初始化模型 (yxdragon 算子化方案)
sess_resize = ort.InferenceSession('resize_trans.onnx')
sess_det    = ort.InferenceSession('ppocrv5_det.onnx')
sess_box    = ort.InferenceSession('bbox_extract.onnx')
sess_line   = ort.InferenceSession('line_extract.onnx')
sess_rec    = ort.InferenceSession('ppocrv5_rec.onnx')
sess_ctc    = ort.InferenceSession('ctc_decode.onnx')

# 2. 图像读取与预处理
img = np.array(Image.open('test.jpg').convert('RGB'))
# 模型内 Resize：输出标准图、半图及缩放系数
imgf, imghalf, scale = sess_resize.run(None, {'image': img, 'dial': np.array(1600, dtype=np.int32)})

# 3. 算子化推理流
hot = sess_det.run(None, {'x': imghalf})[0]
# SVD 提取旋转框：输入热图，直接输出坐标顶点
boxes = sess_box.run(None, {'hotimg': hot[0,0], 'scale': scale, 'thr': 0.3})[0]

for box in boxes:
    # Grid Sample 行提取：直接在大图 Tensor 上抠图校正
    line_tensor, _ = sess_line.run(None, {'x': imgf[0], 'boxes': box, 'scale': scale})
    # 识别与解码
    feat = sess_rec.run(None, {'x': line_tensor})[0]
    indices, prob = sess_ctc.run(None, {'x': feat[0]})
    print(f"识别结果索引: {indices}")
```

## 3. Web 端纯前端 Demo (通过 CDN 加载)
模型体积 < 20MB，支持通过 `jsDelivr` 实现零部署体验。

```javascript
// 核心推理代码
async function runOCR(imageElement) {
    // 利用 jsDelivr 加载轻量化 ONNX 模型
    const modelPath = 'https://cdn.jsdelivr.net/gh/yxdragon/ppocr-onnx@main/model/';
    const detSess = await ort.InferenceSession.create(`${modelPath}ppocrv5_det.onnx`);
    const boxSess = await ort.InferenceSession.create(`${modelPath}bbox_extract.onnx`);
    
    // 1. 获取检测热图
    const { output: hotmap } = await detSess.run({ x: inputTensor });

    // 2. yxdragon 方案：SVD 算子直接解算包围盒
    const { boxes } = await boxSess.run({ 
        hotimg: hotmap, 
        scale: scaleFactor, 
        thr: ort.Tensor.fromFloat32([0.3]) 
    });

    // 3. 坐标直接可用，通过 line_extract.onnx 获取识别行...
}
```

## 4. 如何发布 Release 并使用 CDN
1.  在 GitHub 项目页面点击 **Releases -> Draft a new release**。
2.  将训练好的 `.onnx` 模型文件打包上传。
3.  发布后，即可通过以下格式引用模型：
    `https://cdn.jsdelivr.net/gh/用户名/项目名@版本号/路径/模型文件.onnx`

---
作者：yxdragon (闫霄龙)
GitHub: [https://github.com/yxdragon](https://github.com/yxdragon)