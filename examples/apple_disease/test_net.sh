TEST_IMG=$1
TEST_FILE=$2
build/examples/apple_disease/apple_disease_classification.bin \
models/apple_disease/deploy_resnet50.prototxt \
models/apple_disease/resnet50_train_iter_8000.caffemodel \
examples/apple_disease/mean.binaryproto \
examples/apple_disease/apple_disease_labels.txt \
"${TEST_IMG}" \
$TEST_FILE

