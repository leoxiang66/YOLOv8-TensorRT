export root=${PWD}
cd csrc/detect/end2end
mkdir -p build && cd build
cmake ..
make
mv *.out ${root}
cd ${root}


./yolov8_main1.out yolov8s.engine data/bus.jpg
./yolov8_main2.out yolov8s.engine data/bus.jpg
./yolov8_main3.out yolov8s.engine data/bus.jpg
 