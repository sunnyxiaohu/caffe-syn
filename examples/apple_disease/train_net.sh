./build/tools/caffe train \
    --solver=./models/apple_disease/solver.prototxt \
    --weights=./models/apple_disease/ResNet-50-model.caffemodel \
    2>&1 | tee logs/apple_disease/apple_disease_tranval.log

#    --weights=./models/apple_disease/bvlc_reference_caffenet.caffemodel \
#    --weights=./models/apple_disease/vgg16.caffemodel \
#    --weights=./models/apple_disease/ResNet-50-model.caffemodel \

