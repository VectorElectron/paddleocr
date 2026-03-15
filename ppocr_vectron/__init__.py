import onnxruntime as ort
import numpy as np

root = __file__.replace('\\', '/').rsplit('/', 1)[0] + '/model/'

with open(root + 'ppocrv5_dict.txt', encoding='utf-8') as f:
    lut = np.array(f.read().split('\n') + [' '])

resize_net = ort.InferenceSession(root + 'resize_trans.onnx')
det_net = ort.InferenceSession(root + 'ppocrv5_det.onnx')
box_net = ort.InferenceSession(root + 'bbox_extract.onnx')
extract_net = ort.InferenceSession(root + 'line_extract.onnx')
rec_net = ort.InferenceSession(root + 'ppocrv5_rec.onnx')
ctc_net = ort.InferenceSession(root + 'ctc_decode.onnx')

def ocr(img, dial=None, thr=0.3, boxthr=0.7, sizethr=3, mar=0.5, maxnum=100):
    if dial is None: dial = np.linalg.norm(img.shape[:2])
    
    dial = np.array(dial, dtype=np.int32)
    thr = np.array(thr, dtype=np.float32)
    boxthr = np.array(boxthr, dtype=np.float32)
    sizethr = np.array(sizethr, dtype=np.float32)
    mar = np.array(mar, dtype=np.float32)
    maxnum = np.array(maxnum, dtype=np.int32)
    
    imgf, imghalf, scale = resize_net.run(None, {'image':img, 'dial':dial})
    hot = det_net.run(None, {'x':imghalf})[0]
    para = {'hotimg': hot[0,0], 'scale':scale, 'thr':thr, 'boxthr':boxthr,
        'sizethr':sizethr, 'mar':mar, 'maxn':maxnum}
    boxes = box_net.run(None, para)[0]
    blocks, widths = extract_net.run(None, {'x':imgf[0], 'boxes':boxes, 'scale':scale})

    result = []
    for box, im, w in zip(boxes, blocks, widths):
        line = rec_net.run(None, {'x':im[None, :, :, :w]})[0]
        cont, prob = ctc_net.run(None, {'x': line[0]})
        cont = ''.join([lut[i-1] for i in cont])
        if prob < 0.6: continue
        result.append((box, cont, prob))
    return result

def test():
    from imageio.v2 import imread, imsave
    import matplotlib.pyplot as plt

    print('\n识别结果：')
    img = imread(root + 'testimg.png')
    result = ocr(img, mar=0.6)
    for i in result: print(i[1])
    import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.imshow(img)
    for box, cont, prob in result:
        plt.plot(*(box).T, 'blue')
        plt.text(*box[0], cont+':%.2f'%prob, color='red')
    plt.show()

if __name__ == '__main__':
    test()
