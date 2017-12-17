TOOLS=build/tools
DATA=examples/apple_disease/apple_disease_train_lmdb
OUT=examples/apple_disease

$TOOLS/compute_image_mean $DATA $OUT/mean.binaryproto

