import argparse
from io import BytesIO

import onnx
import torch
from ultralytics import YOLO
from torch.quantization import QuantStub, DeQuantStub, fuse_modules

from models.common import PostDetect, optim

try:
    import onnxsim
except ImportError:
    onnxsim = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w',
                        '--weights',
                        type=str,
                        required=True,
                        help='PyTorch yolov8 weights')
    parser.add_argument('--iou-thres',
                        type=float,
                        default=0.65,
                        help='IOU threshoud for NMS plugin')
    parser.add_argument('--conf-thres',
                        type=float,
                        default=0.25,
                        help='CONF threshoud for NMS plugin')
    parser.add_argument('--topk',
                        type=int,
                        default=100,
                        help='Max number of detection bboxes')
    parser.add_argument('--opset',
                        type=int,
                        default=11,
                        help='ONNX opset version')
    parser.add_argument('--sim',
                        action='store_true',
                        help='simplify onnx model')
    parser.add_argument('--input-shape',
                        nargs='+',
                        type=int,
                        default=[1, 3, 640, 640],
                        help='Model input shape only for api builder')
    parser.add_argument('--device',
                        type=str,
                        default='cpu',
                        help='Export ONNX device')
    args = parser.parse_args()
    assert len(args.input_shape) == 4
    PostDetect.conf_thres = args.conf_thres
    PostDetect.iou_thres = args.iou_thres
    PostDetect.topk = args.topk
    return args


def main(args):
    b = args.input_shape[0]
    YOLOv8 = YOLO(args.weights)
    model = YOLOv8.model.fuse().eval()
    
    # Step 1: 添加 QuantStub 和 DeQuantStub
    model.quant = QuantStub()
    model.dequant = DeQuantStub()
    
    # Step 2: 融合模块
    for m in model.modules():
        if isinstance(m, torch.nn.Sequential):
            if len(m) == 2 and isinstance(m[0], torch.nn.Conv2d) and isinstance(m[1], torch.nn.BatchNorm2d):
                fuse_modules(m, ['0', '1'], inplace=True)
            elif len(m) == 3 and isinstance(m[0], torch.nn.Conv2d) and isinstance(m[1], torch.nn.BatchNorm2d) and isinstance(m[2], torch.nn.ReLU):
                fuse_modules(m, ['0', '1', '2'], inplace=True)

    # Step 3: 指定量化配置并准备模型量化
    qconfig = torch.quantization.QConfig(
        activation=torch.quantization.MinMaxObserver.with_args(dtype=torch.quint8),
        weight=torch.quantization.MinMaxObserver.with_args(dtype=torch.qint8)
    )
    torch.quantization.prepare(model, inplace=True)
    
    # Step 4: 校准模型
    for m in model.modules():
        optim(m)
        m.to(args.device)
    model.to(args.device)
    fake_input = torch.randn(args.input_shape).to(args.device)
    model(fake_input)
        
    # Step 5: 转换量化模型
    torch.quantization.convert(model, inplace=True)
    
    save_path = args.weights.replace('.pt', '_quant.onnx')
    with BytesIO() as f:
        torch.onnx.export(
            model,
            fake_input,
            f,
            opset_version=args.opset,
            input_names=['images'],
            output_names=['num_dets', 'bboxes', 'scores', 'labels'])
        f.seek(0)
        onnx_model = onnx.load(f)
    onnx.checker.check_model(onnx_model)
    shapes = [b, 1, b, args.topk, 4, b, args.topk, b, args.topk]
    for i in onnx_model.graph.output:
        for j in i.type.tensor_type.shape.dim:
            j.dim_param = str(shapes.pop(0))
    if args.sim:
        try:
            onnx_model, check = onnxsim.simplify(onnx_model)
            assert check, 'assert check failed'
        except Exception as e:
            print(f'Simplifier failure: {e}')
    onnx.save(onnx_model, save_path)
    print(f'ONNX export success, saved as {save_path}')


if __name__ == '__main__':
    main(parse_args())