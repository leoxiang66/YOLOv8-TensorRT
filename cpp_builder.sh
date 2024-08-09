# Prompt user for Python and pip paths
echo "请输入Python路径:"
read python_path

echo "请输入pip路径:"
read pip_path

# Install python environments
echo "安装python环境 ..."
cat ./requirements.txt
$pip_path install -r requirements.txt

echo ""
echo ""

echo "导出yolov8s onnx ..."
$python_path export-det.py \
--weights yolov8s.pt \
--iou-thres 0.65 \
--conf-thres 0.25 \
--topk 100 \
--opset 11 \
--sim \
--input-shape 1 3 640 640 \
--device cuda:0

echo ""
echo ""

echo "转为enginefile ..."
# /usr/local/TensorRT/bin/trtexec --onnx=yolov8s.onnx --saveEngine=yolov8s.engine --fp16
$python_path build.py \
--weights yolov8s.onnx \
--iou-thres 0.65 \
--conf-thres 0.25 \
--topk 100 \
--fp16  \
--device cuda:0
echo ""
echo ""