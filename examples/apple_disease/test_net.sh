TEST_IMG=$1
TEST_FILE=$2
for i in $(seq 1 50)
do

build-cpu/examples/apple_disease/apple_disease_classification \
models/apple_disease/deploy_resnet50.prototxt \
models/apple_disease/resnet50_train_iter_1000.caffemodel \
models/apple_disease/mean.binaryproto \
models/apple_disease/apple_disease_labels.txt \
"${TEST_IMG}" \
$TEST_FILE

done

