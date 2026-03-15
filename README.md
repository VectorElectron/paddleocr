# PaddleOCR Vectron

A lightweight, operator-based OCR (Optical Character Recognition) engine designed for cross-platform and cross-language deployment.

## Key Features

### 🚀 Full Operator-based Pipeline
PaddleOCR Vectron implements the entire OCR process through a series of specialized operators, each optimized for their specific task:

- **Resize Operator**: Efficiently resizes images for optimal processing
- **Detection Operator**: Locates text regions in images
- **Bounding Box Extraction**: Precisely extracts text-containing regions
- **Line Extraction**: Separates text into individual lines
- **Recognition Operator**: Identifies characters within each line
- **CTC Decoder**: Converts model outputs to readable text

This modular design allows for easy customization and optimization of each component.



### 🔄 Cross-Language & Edge Support
The ONNX-based architecture enables versatile deployment:

- **Multiple Languages**: Python (native support), C++, C#, Java, JavaScript, and more via ONNX runtime bindings
- **Edge Acceleration**: Supports TensorRT for NVIDIA devices and RKNN for Rockchip edge devices
- **Cross-Platform**: Runs seamlessly on Windows, Linux, macOS, mobile devices, and edge devices

### 📦 Lightweight Design
- Minimal dependencies (only ONNX runtime and NumPy)
- Compact model files
- Efficient inference with ONNX runtime optimization

## Installation

### From PyPI
```bash
pip install ppocr-vectron
```

### From Source
```bash
pip install -e .
```

### Package Location
After pip installation, the package is located at `site-packages/ppocr_vectron/`.

### Model Files
After installation, you need to download the model and dictionary files from the [release page](https://github.com/VectorElectron/release/releases) and place them in the `ppocr_vectron/model/` directory.

Required files:
- `ppocrv5_dict.txt`
- `ppocrv5_det.onnx`
- `ppocrv5_rec.onnx`
- `resize_trans.onnx`
- `bbox_extract.onnx`
- `line_extract.onnx`
- `ctc_decode.onnx`

## Python Usage

After installing the package and downloading the model files to the `model/` directory, you can use PaddleOCR Vectron as follows:

```python
import numpy as np
from imageio.v2 import imread
from ppocr_vectron import ocr
 
# Load an image
image = imread('path/to/your/image.png')
 
# Run OCR with default parameters
results = ocr(image)
 
# Print results
print("OCR Results:")
for box, text, confidence in results:
    print(f"Text: {text}")
    print(f"Confidence: {confidence:.2f}")
    print(f"Bounding Box: {box}")
    print("-")
 
# Advanced usage with custom parameters
custom_results = ocr(
    image,
    dial=1000,      # Maximum dimension for resizing
    thr=0.3,        # Detection threshold
    boxthr=0.7,     # Bounding box threshold
    sizethr=3,      # Minimum size threshold
    mar=0.5,        # Margin around text
    maxnum=100      # Maximum number of text regions
)
```
![Image](https://github.com/user-attachments/assets/cfec2323-0202-416b-b000-f2dae6937629)

## JavaScript Usage

PaddleOCR Vectron supports pure frontend deployment using ONNX Runtime Web. For a complete implementation, please refer to `model/index-scan.html`.

### Live Demo
Try the complete mobile-compatible pure frontend OCR experience at:
[https://vectorelectron.github.io/release/paddleocr/index-scan.html](https://vectorelectron.github.io/release/paddleocr/index-scan.html)

### Quick Start

```html
<!DOCTYPE html>
<html>
<head>
    <title>PPOCR Vectron JS Example</title>
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.0/dist/ort.min.js"></script>
</head>
<body>
    <input type="file" id="imageInput" accept="image/*">
    <button onclick="runOCR()">Run OCR</button>
    <div id="results"></div>

    <script>
        let sessions = {};
        let lut = [];

        // Initialize models (simplified)
        async function init() {
            const basePath = './model/'; // Path to model files
            
            // Load dictionary
            const dictResp = await fetch(basePath + 'ppocrv5_dict.txt');
            const text = await dictResp.text();
            lut = text.split('\n').concat([' ']);

            // Load models
            const opt = { executionProviders: ['wasm'] };
            sessions.resize = await ort.InferenceSession.create(basePath + 'resize_trans.onnx', opt);
            sessions.det = await ort.InferenceSession.create(basePath + 'ppocrv5_det.onnx', opt);
            sessions.box = await ort.InferenceSession.create(basePath + 'bbox_extract.onnx', opt);
            sessions.extract = await ort.InferenceSession.create(basePath + 'line_extract.onnx', opt);
            sessions.rec = await ort.InferenceSession.create(basePath + 'ppocrv5_rec.onnx', opt);
            sessions.ctc = await ort.InferenceSession.create(basePath + 'ctc_decode.onnx', opt);

            console.log('Models loaded successfully');
        }

        // Run OCR (simplified)
        async function runOCR() {
            // Full implementation available in model/index-scan.html
            // See the live demo for complete functionality
            console.log('Running OCR...');
            // Implementation details omitted - refer to index-scan.html
        }

        // Initialize on page load
        window.onload = init;
    </script>
</body>
</html>
```

For the complete implementation with image processing, parallel execution, and UI components, please refer to the full code in `model/index-scan.html`.

![Image](https://github.com/user-attachments/assets/8a6ab9d7-81e3-4685-a5fc-51b1c616587a)

## License
BSD 3-Clause License