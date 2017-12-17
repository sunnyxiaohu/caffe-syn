#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV

#include <string>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template<typename Dtype>
DataTransformer<Dtype>::DataTransformer(const TransformationParameter& param,
    Phase phase)
    : param_(param), phase_(phase) {
  // check if we want to use mean_file
  if (param_.has_mean_file()) {
    CHECK_EQ(param_.mean_value_size(), 0) <<
      "Cannot specify mean_file and mean_value at the same time";
    const string& mean_file = param.mean_file();
    if (Caffe::root_solver()) {
      LOG(INFO) << "Loading mean file from: " << mean_file;
    }
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    data_mean_.FromProto(blob_proto);
  }
  // check if we want to use mean_value
  if (param_.mean_value_size() > 0) {
    CHECK(param_.has_mean_file() == false) <<
      "Cannot specify mean_file and mean_value at the same time";
    for (int c = 0; c < param_.mean_value_size(); ++c) {
      mean_values_.push_back(param_.mean_value(c));
    }
  }

  //load multiscale info
  max_distort_ = param_.max_distort();
  custom_scale_ratios_.clear();
  for (int i = 0; i < param_.scale_ratios_size(); ++i){
    custom_scale_ratios_.push_back(param_.scale_ratios(i));
  }
  org_size_proc_ = param.original_image();
}

/** @build fixed crop offsets for random selection
 */
void fillFixOffset(int datum_height, int datum_width, int crop_height, int crop_width,
                   bool more_crop,
                   vector<pair<int , int> >& offsets){
  int height_off = (datum_height - crop_height)/4;
  int width_off = (datum_width - crop_width)/4;

  offsets.clear();
  offsets.push_back(pair<int, int>(0, 0)); //upper left
  offsets.push_back(pair<int, int>(0, 4 * width_off)); //upper right
  offsets.push_back(pair<int, int>(4 * height_off, 0)); //lower left
  offsets.push_back(pair<int, int>(4 * height_off, 4 *width_off)); //lower right
  offsets.push_back(pair<int, int>(2 * height_off, 2 * width_off)); //center

  //will be used when more_fix_crop is set to true
  if (more_crop) {
    offsets.push_back(pair<int, int>(0, 2 * width_off)); //top center
    offsets.push_back(pair<int, int>(4 * height_off, 2 * width_off)); //bottom center
    offsets.push_back(pair<int, int>(2 * height_off, 0)); //left center
    offsets.push_back(pair<int, int>(2 * height_off, 4 * width_off)); //right center

    offsets.push_back(pair<int, int>(1 * height_off, 1 * width_off)); //upper left quarter
    offsets.push_back(pair<int, int>(1 * height_off, 3 * width_off)); //upper right quarter
    offsets.push_back(pair<int, int>(3 * height_off, 1 * width_off)); //lower left quarter
    offsets.push_back(pair<int, int>(3 * height_off, 3 * width_off)); //lower right quarter
  }
}

float _scale_rates[] = {1.0, .875, .75, .66};
vector<float> default_scale_rates(_scale_rates, _scale_rates + sizeof(_scale_rates)/ sizeof(_scale_rates[0]) );

/**
 * @generate crop size when multi-scale cropping is requested
 */
void fillCropSize(int input_height, int input_width,
                 int net_input_height, int net_input_width,
                 vector<pair<int, int> >& crop_sizes,
                 int max_distort, vector<float>& custom_scale_ratios){
    crop_sizes.clear();

    vector<float>& scale_rates = (custom_scale_ratios.size() > 0)?custom_scale_ratios:default_scale_rates;
    int base_size = std::min(input_height, input_width);
    for (int h = 0; h < scale_rates.size(); ++h){
      int crop_h = int(base_size * scale_rates[h]);
      crop_h = (abs(crop_h - net_input_height) < 3)?net_input_height:crop_h;
      for (int w = 0; w < scale_rates.size(); ++w){
        int crop_w = int(base_size * scale_rates[w]);
        crop_w = (abs(crop_w - net_input_width) < 3)?net_input_width:crop_w;

        //append this cropping size into the list
        if (abs(h-w)<=max_distort) {
          crop_sizes.push_back(pair<int, int>(crop_h, crop_w));
        }
      }
    }
}

/**
 * @generate crop size and offset when process original images
 */
void sampleRandomCropSize(int img_height, int img_width,
                          int& crop_height, int& crop_width,
                          float min_scale=0.08, float max_scale=1.0, float min_as=0.75, float max_as=1.33){
  float total_area = img_height * img_width;
  float area_ratio = 0;
  float target_area = 0;
  float aspect_ratio = 0;
  float flip_coin = 0;

  int attempt = 0;

  while (attempt < 10) {
    // sample scale and area
    caffe_rng_uniform(1, min_scale, max_scale, &area_ratio);
    target_area = total_area * area_ratio;

    caffe_rng_uniform(1, float(0), float(1), &flip_coin);
    if (flip_coin > 0.5){
        std::swap(crop_height, crop_width);
    }

    // sample aspect ratio
    caffe_rng_uniform(1, min_as, max_as, &aspect_ratio);
    crop_height = int(sqrt(target_area / aspect_ratio));
    crop_width = int(sqrt(target_area * aspect_ratio));

    if (crop_height <= img_height && crop_width <= img_width){
      return;
    }
    attempt ++;
  }

  // fallback to normal 256-224 style size crop
  crop_height = img_height / 8 * 7;
  crop_width = img_width / 8 * 7;
}

/**
 * @generate erase size when process images
 */
void fillEraseSize(const int datum_height, const int datum_width, const float min_scale, const float max_scale, 
                   const float min_aspect, const float max_aspect, vector<float>& erase_size){
  float erase_x_min = datum_width, erase_x_max = -1, erase_y_min = datum_height, erase_y_max = -1;
  erase_size.clear();
  int attempt = 0;
  while (attempt < 10) {
      float erase_scale;
      caffe_rng_uniform(1, min_scale, max_scale, &erase_scale);
      float erase_width = (float)datum_width * erase_scale;
      float erase_aspect;
      caffe_rng_uniform(1, min_aspect, max_aspect, &erase_aspect);
      float erase_height = erase_width * erase_aspect;

      caffe_rng_uniform(1, float(0), float(datum_width), &erase_x_min);
      caffe_rng_uniform(1, float(0), float(datum_height), &erase_y_min);
      erase_x_max = erase_x_min + erase_width - 1;
      erase_y_max = erase_y_min + erase_height - 1;
      // LOG(INFO) <<"erase_x_min: "<<erase_x_min<<", erase_y_min: "<<erase_y_min<<", erase_x_max: "<<erase_x_max<<", erase_y_max: "<<erase_y_max;
      if (erase_x_min >= 0 && erase_y_min >= 0 && erase_x_max < datum_width && erase_y_max < datum_height) {
         erase_size.push_back(erase_x_min);
         erase_size.push_back(erase_y_min);
         erase_size.push_back(erase_x_max);
         erase_size.push_back(erase_y_max);
         return;
      }

      attempt ++;
  }
  
  erase_size.push_back(erase_x_min);
  erase_size.push_back(erase_y_min);
  erase_size.push_back(erase_x_max);
  erase_size.push_back(erase_y_max);
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum& datum,
                                       Dtype* transformed_data) {
  const string& data = datum.data();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  const int crop_size = param_.crop_size();
  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_uint8 = data.size() > 0;
  const bool has_mean_values = mean_values_.size() > 0;

  CHECK_GT(datum_channels, 0);
  CHECK_GE(datum_height, crop_size);
  CHECK_GE(datum_width, crop_size);

  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(datum_channels, data_mean_.channels());
    CHECK_EQ(datum_height, data_mean_.height());
    CHECK_EQ(datum_width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == datum_channels) <<
     "Specify either 1 mean_value or as many as channels: " << datum_channels;
    if (datum_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < datum_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  int height = datum_height;
  int width = datum_width;

  int h_off = 0;
  int w_off = 0;
  if (crop_size) {
    height = crop_size;
    width = crop_size;
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(datum_height - crop_size + 1);
      w_off = Rand(datum_width - crop_size + 1);
    } else {
      h_off = (datum_height - crop_size) / 2;
      w_off = (datum_width - crop_size) / 2;
    }
  }

  Dtype datum_element;
  int top_index, data_index;
  for (int c = 0; c < datum_channels; ++c) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        data_index = (c * datum_height + h_off + h) * datum_width + w_off + w;
        if (do_mirror) {
          top_index = (c * height + h) * width + (width - 1 - w);
        } else {
          top_index = (c * height + h) * width + w;
        }
        if (has_uint8) {
          datum_element =
            static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
        } else {
          datum_element = datum.float_data(data_index);
        }
        if (has_mean_file) {
          transformed_data[top_index] =
            (datum_element - mean[data_index]) * scale;
        } else {
          if (has_mean_values) {
            transformed_data[top_index] =
              (datum_element - mean_values_[c]) * scale;
          } else {
            transformed_data[top_index] = datum_element * scale;
          }
        }
      }
    }
  }
}


template<typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum& datum,
                                       Blob<Dtype>* transformed_blob) {
  // If datum is encoded, decoded and transform the cv::image.
  if (datum.encoded()) {
#ifdef USE_OPENCV
    CHECK(!(param_.force_color() && param_.force_gray()))
        << "cannot set both force_color and force_gray";
    cv::Mat cv_img;
    if (param_.force_color() || param_.force_gray()) {
    // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    // Transform the cv::image into blob.
    return Transform(cv_img, transformed_blob);
#else
    LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  } else {
    if (param_.force_color() || param_.force_gray()) {
      LOG(ERROR) << "force_color and force_gray only for encoded datum";
    }
  }

  const int crop_size = param_.crop_size();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  // Check dimensions.
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int num = transformed_blob->num();

  CHECK_EQ(channels, datum_channels);
  CHECK_LE(height, datum_height);
  CHECK_LE(width, datum_width);
  CHECK_GE(num, 1);

  if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
  } else {
    CHECK_EQ(datum_height, height);
    CHECK_EQ(datum_width, width);
  }

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
  Transform(datum, transformed_data);
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const vector<Datum> & datum_vector,
                                       Blob<Dtype>* transformed_blob) {
  const int datum_num = datum_vector.size();
  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();

  CHECK_GT(datum_num, 0) << "There is no datum to add";
  CHECK_LE(datum_num, num) <<
    "The size of datum_vector must be no greater than transformed_blob->num()";
  Blob<Dtype> uni_blob(1, channels, height, width);
  for (int item_id = 0; item_id < datum_num; ++item_id) {
    int offset = transformed_blob->offset(item_id);
    uni_blob.set_cpu_data(transformed_blob->mutable_cpu_data() + offset);
    Transform(datum_vector[item_id], &uni_blob);
  }
}

#ifdef USE_OPENCV
template<typename Dtype>
void DataTransformer<Dtype>::Transform(const vector<cv::Mat> & mat_vector,
                                       Blob<Dtype>* transformed_blob,
                                       const bool is_video) {
  if (is_video) {
    const int mat_num = mat_vector.size();
    const int num = transformed_blob->shape(0);
    const int channels = transformed_blob->shape(1);
    const int length = transformed_blob->shape(2);
    const int height = transformed_blob->shape(3);
    const int width = transformed_blob->shape(4);
    const int img_height = mat_vector[0].rows;
    const int img_width = mat_vector[0].cols;

    // mirror / cropping is picked once here, and will be reused for all frames
    // within a video clip
    const bool rand_mirror = param_.mirror()
                             ? static_cast<bool>(Rand(2)) : false;
    const int crop_size = param_.crop_size();
    const int rand_h_off = (phase_ == TRAIN && param_.crop_size())
                           ? Rand(img_height - crop_size + 1) : 0;
    const int rand_w_off = (phase_ == TRAIN && param_.crop_size())
                           ? Rand(img_width - crop_size + 1) : 0;
    float tmp, rotate_off; 
    caffe_rng_uniform(1, 0.0f, 1.0f, &tmp);
    caffe_rng_uniform(1, param_.rotate_min(), param_.rotate_max(), &rotate_off);
    bool do_rotate = param_.has_rotate_ratio() && tmp < param_.rotate_ratio() && rotate_off!=0;
    rotate_off = do_rotate ? rotate_off : 0;
    caffe_rng_uniform(1, 0.0f, 1.0f, &tmp);
    bool do_erase = param_.has_erase_ratio() && tmp < param_.erase_ratio();
    vector<float> erase_off; //x_min, y_min, x_max, y_max
    if (do_erase) {
      fillEraseSize(height, width, param_.scale_min(), param_.scale_max(), 
                   param_.aspect_min(), param_.aspect_max(), erase_off);
      do_erase = erase_off[0] >= 0 && erase_off[1] >= 0 && erase_off[2] < width && erase_off[3] < height;
      if (!do_erase) {
         erase_off.clear();
      } 
    }
    CHECK_GT(mat_num, 0) << "There is no MAT to add";
    CHECK_EQ(num, 1) << "First dimension (batch number) must be 1";
    CHECK_EQ(mat_num, length) <<
      "The size of mat_vector must be equals to transformed_blob->shape(2)";
    vector<int> uni_blob_shape(5);
    uni_blob_shape[0] = 1;
    uni_blob_shape[1] = channels;
    uni_blob_shape[2] = 1;
    uni_blob_shape[3] = height;
    uni_blob_shape[4] = width;
    Blob<Dtype> uni_blob(uni_blob_shape);
    int offset;
    vector<int> transformed_blob_offset(5, 0);
    for (int item_id = 0; item_id < mat_num; ++item_id) {
      Transform(mat_vector[item_id],
                &uni_blob,
                is_video,
                item_id,
                rand_mirror,
                rand_h_off,
                rand_w_off,
                rotate_off,
                erase_off);
      transformed_blob_offset[2] = item_id;
      for (int c = 0; c < channels; ++c) {
        transformed_blob_offset[1] = c;
        offset = transformed_blob->offset(transformed_blob_offset);
        caffe_copy(height * width, uni_blob.cpu_data() + c * height * width,
                   transformed_blob->mutable_cpu_data() + offset);
      }
    }
  } else {
    const int mat_num = mat_vector.size();
    const int num = transformed_blob->num();
    const int channels = transformed_blob->channels();
    const int height = transformed_blob->height();
    const int width = transformed_blob->width();

    CHECK_GT(mat_num, 0) << "There is no MAT to add";
    CHECK_EQ(mat_num, num) <<
      "The size of mat_vector must be equals to transformed_blob->num()";
    Blob<Dtype> uni_blob(1, channels, height, width);
    for (int item_id = 0; item_id < mat_num; ++item_id) {
      int offset = transformed_blob->offset(item_id);
      uni_blob.set_cpu_data(transformed_blob->mutable_cpu_data() + offset);
      Transform(mat_vector[item_id], &uni_blob, false);
    }
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const cv::Mat& cv_img,
                                       Blob<Dtype>* transformed_blob,
                                       const bool is_video,
                                       const int frame,
                                       const bool rand_mirror,
                                       const int rand_h_off,
                                       const int rand_w_off,
                                       const float rotate_off,
                                       const vector<float>& erase_off) {
  const int crop_size = param_.crop_size();
  const int img_channels = cv_img.channels();
  const int img_height = cv_img.rows;
  const int img_width = cv_img.cols;

  // Check dimensions.
  const int channels = transformed_blob->shape(1);
  const int height = transformed_blob->shape(is_video ? 3 : 2);
  const int width = transformed_blob->shape(is_video ? 4 : 3);
  const int num = transformed_blob->shape(0);

  CHECK_EQ(channels, img_channels);
  CHECK_LE(height, img_height);
  CHECK_LE(width, img_width);
  CHECK_GE(num, 1);

  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";

  const Dtype scale = param_.scale();
  const bool do_mirror = is_video ?
                         param_.mirror() && rand_mirror :
                         param_.mirror() && Rand(2);
  float tmp, angle; 
  caffe_rng_uniform(1, 0.0f, 1.0f, &tmp);
  caffe_rng_uniform(1, param_.rotate_min(), param_.rotate_max(), &angle);
  bool do_rotate = param_.has_rotate_ratio() && tmp < param_.rotate_ratio() && angle!=0;
  do_rotate = is_video ? (rotate_off!=0) : do_rotate;
  angle = is_video ? rotate_off : angle;
  caffe_rng_uniform(1, 0.0f, 1.0f, &tmp);
  bool do_erase = param_.has_erase_ratio() && tmp < param_.erase_ratio();
  do_erase = is_video ? (!erase_off.empty()) : do_erase;
  vector<float> erase_size; //x_min, y_min, x_max, y_max
  if (do_erase) {
    if (is_video) {
       erase_size.assign(erase_off.begin(), erase_off.end());
    } else {
      fillEraseSize(height, width, param_.scale_min(), param_.scale_max(), 
                   param_.aspect_min(), param_.aspect_max(), erase_size);
    }
    do_erase = erase_size[0] >= 0 && erase_size[1] >= 0 && erase_size[2] < width && erase_size[3] < height;
  }
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;
  const bool is_mean_cube = data_mean_.shape().size() == 5;
  unsigned int length = 0;

  CHECK_GT(img_channels, 0);
  CHECK_GE(img_height, crop_size);
  CHECK_GE(img_width, crop_size);

  Dtype* mean = NULL;
  if (has_mean_file) {
    if (is_mean_cube) {
      CHECK_EQ(img_channels, data_mean_.shape(1));
      length = data_mean_.shape(2);
      CHECK_LE(frame, length) << "frame number=" << frame << " must be less "
                              << "or equal to length=" << length;
      CHECK_EQ(img_height, data_mean_.shape(3));
      CHECK_EQ(img_width, data_mean_.shape(4));
    } else {
      CHECK_EQ(img_channels, data_mean_.channels());
      CHECK_EQ(img_height, data_mean_.height());
      CHECK_EQ(img_width, data_mean_.width());
    }
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    //CHECK(mean_values_.size() == 1 || mean_values_.size() == img_channels) <<
    // "Specify either 1 mean_value or as many as channels: " << img_channels;
    if (img_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < img_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  int h_off = 0;
  int w_off = 0;
  cv::Mat cv_cropped_img = cv_img;
  if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      if (is_video) {
        h_off = rand_h_off;
        w_off = rand_w_off;
      } else {
        h_off = Rand(img_height - crop_size + 1);
        w_off = Rand(img_width - crop_size + 1);
      }
    } else {
      h_off = (img_height - crop_size) / 2;
      w_off = (img_width - crop_size) / 2;
    }
    cv::Rect roi(w_off, h_off, crop_size, crop_size);
    cv_cropped_img = cv_img(roi);
  } else {
    CHECK_EQ(img_height, height);
    CHECK_EQ(img_width, width);
  }

  CHECK(cv_cropped_img.data);
  
  if (do_rotate) {
     cv::Point2f center(width/2, height/2);
     cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1);
     cv::Rect bbox = cv::RotatedRect(center, cv_cropped_img.size(), angle).boundingRect();

     rot.at<double>(0, 2) += bbox.width / 2.0 - center.x;
     rot.at<double>(1, 2) += bbox.height / 2.0 - center.y;
     cv::warpAffine(cv_cropped_img, cv_cropped_img, rot, bbox.size());
  }
  
  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
  int top_index;
  for (int h = 0; h < height; ++h) {
    const uchar* ptr = cv_cropped_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < img_channels; ++c) {
        if (do_mirror) {
          top_index = (c * height + h) * width + (width - 1 - w);
        } else {
          top_index = (c * height + h) * width + w;
        }
        Dtype pixel;
        if (do_erase && w >= erase_size[0] && w <= erase_size[2] && h >= erase_size[1] && h <= erase_size[3] ) {
            pixel = Rand(255);
        } else {
            pixel = static_cast<Dtype>(ptr[img_index]);
        }
        img_index ++;
        if (has_mean_file) {
          int mean_index = is_mean_cube ? ((c * length + frame) * img_height
                                           + h_off + h) * img_width + w_off
                                           + w
                                        : (c * img_height + h_off + h)
                                          * img_width + w_off + w;
          transformed_data[top_index] = (pixel - mean[mean_index]) * scale;
        } else if (has_mean_values) {
            transformed_data[top_index] =
              (pixel - mean_values_[c]) * scale;
        } else {
          transformed_data[top_index] = pixel * scale;
        }
      }
    }
  }
}
#endif  // USE_OPENCV

template<typename Dtype>
void DataTransformer<Dtype>::Transform(Blob<Dtype>* input_blob,
                                       Blob<Dtype>* transformed_blob) {
  const int crop_size = param_.crop_size();
  const int input_num = input_blob->num();
  const int input_channels = input_blob->channels();
  const int input_height = input_blob->height();
  const int input_width = input_blob->width();

  if (transformed_blob->count() == 0) {
    // Initialize transformed_blob with the right shape.
    if (crop_size) {
      transformed_blob->Reshape(input_num, input_channels,
                                crop_size, crop_size);
    } else {
      transformed_blob->Reshape(input_num, input_channels,
                                input_height, input_width);
    }
  }

  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int size = transformed_blob->count();

  CHECK_LE(input_num, num);
  CHECK_EQ(input_channels, channels);
  CHECK_GE(input_height, height);
  CHECK_GE(input_width, width);


  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  int h_off = 0;
  int w_off = 0;
  if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(input_height - crop_size + 1);
      w_off = Rand(input_width - crop_size + 1);
    } else {
      h_off = (input_height - crop_size) / 2;
      w_off = (input_width - crop_size) / 2;
    }
  } else {
    CHECK_EQ(input_height, height);
    CHECK_EQ(input_width, width);
  }

  Dtype* input_data = input_blob->mutable_cpu_data();
  if (has_mean_file) {
    CHECK_EQ(input_channels, data_mean_.channels());
    CHECK_EQ(input_height, data_mean_.height());
    CHECK_EQ(input_width, data_mean_.width());
    for (int n = 0; n < input_num; ++n) {
      int offset = input_blob->offset(n);
      caffe_sub(data_mean_.count(), input_data + offset,
            data_mean_.cpu_data(), input_data + offset);
    }
  }

  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == input_channels) <<
     "Specify either 1 mean_value or as many as channels: " << input_channels;
    if (mean_values_.size() == 1) {
      caffe_add_scalar(input_blob->count(), -(mean_values_[0]), input_data);
    } else {
      for (int n = 0; n < input_num; ++n) {
        for (int c = 0; c < input_channels; ++c) {
          int offset = input_blob->offset(n, c);
          caffe_add_scalar(input_height * input_width, -(mean_values_[c]),
            input_data + offset);
        }
      }
    }
  }

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();

  for (int n = 0; n < input_num; ++n) {
    int top_index_n = n * channels;
    int data_index_n = n * channels;
    for (int c = 0; c < channels; ++c) {
      int top_index_c = (top_index_n + c) * height;
      int data_index_c = (data_index_n + c) * input_height + h_off;
      for (int h = 0; h < height; ++h) {
        int top_index_h = (top_index_c + h) * width;
        int data_index_h = (data_index_c + h) * input_width + w_off;
        if (do_mirror) {
          int top_index_w = top_index_h + width - 1;
          for (int w = 0; w < width; ++w) {
            transformed_data[top_index_w - w] = input_data[data_index_h + w];
          }
        } else {
          for (int w = 0; w < width; ++w) {
            transformed_data[top_index_h + w] = input_data[data_index_h + w];
          }
        }
      }
    }
  }
  if (scale != Dtype(1)) {
    DLOG(INFO) << "Scale: " << scale;
    caffe_scal(size, scale, transformed_data);
  }
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(const Datum& datum) {
  if (datum.encoded()) {
#ifdef USE_OPENCV
    CHECK(!(param_.force_color() && param_.force_gray()))
        << "cannot set both force_color and force_gray";
    cv::Mat cv_img;
    if (param_.force_color() || param_.force_gray()) {
    // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    // InferBlobShape using the cv::image.
    return InferBlobShape(cv_img);
#else
    LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  }
  const int crop_size = param_.crop_size();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();
  // Check dimensions.
  CHECK_GT(datum_channels, 0);
  CHECK_GE(datum_height, crop_size);
  CHECK_GE(datum_width, crop_size);
  // Build BlobShape.
  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = datum_channels;
  shape[2] = (crop_size)? crop_size: datum_height;
  shape[3] = (crop_size)? crop_size: datum_width;
  return shape;
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(
    const vector<Datum> & datum_vector) {
  const int num = datum_vector.size();
  CHECK_GT(num, 0) << "There is no datum to in the vector";
  // Use first datum in the vector to InferBlobShape.
  vector<int> shape = InferBlobShape(datum_vector[0]);
  // Adjust num to the size of the vector.
  shape[0] = num;
  return shape;
}

#ifdef USE_OPENCV
template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(const cv::Mat& cv_img) {
  const int crop_size = param_.crop_size();
  const int img_channels = cv_img.channels();
  const int img_height = cv_img.rows;
  const int img_width = cv_img.cols;
  // Check dimensions.
  CHECK_GT(img_channels, 0);
  CHECK_GE(img_height, crop_size);
  CHECK_GE(img_width, crop_size);
  // Build BlobShape.
  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = img_channels;
  shape[2] = (crop_size)? crop_size: img_height;
  shape[3] = (crop_size)? crop_size: img_width;
  return shape;
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(
    const vector<cv::Mat> & mat_vector, const bool is_video) {
  const int num = mat_vector.size();
  CHECK_GT(num, 0) << "There is no cv_img to in the vector";
  if (is_video) {
    vector<int> tmp_shape = InferBlobShape(mat_vector, false);
    CHECK_EQ(tmp_shape.size(), 4) << "A mat_vector must be 4-dimensional";
    vector<int> shape(5);
    shape[0] = 1;             // num of batches
    shape[1] = tmp_shape[1];  // num of channels
    shape[2] = num;           // this is actually "length" of C3D blob
    shape[3] = tmp_shape[2];
    shape[4] = tmp_shape[3];
    return shape;
  } else {
    // Use first cv_img in the vector to InferBlobShape.
    vector<int> shape = InferBlobShape(mat_vector[0]);
    // Adjust num to the size of the vector.
    shape[0] = num;
    return shape;
  }
}
#endif  // USE_OPENCV

template <typename Dtype>
void DataTransformer<Dtype>::InitRand() {
  const bool needs_rand = param_.mirror() ||
      (phase_ == TRAIN && param_.crop_size());
  if (needs_rand) {
    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rng_seed));
  } else {
    rng_.reset();
  }
}

template <typename Dtype>
void DataTransformer<Dtype>::SetRandFromSeed(const unsigned int rng_seed) {
  rng_.reset(new Caffe::RNG(rng_seed));
}

template <typename Dtype>
int DataTransformer<Dtype>::Rand(int n) {
  CHECK(rng_);
  CHECK_GT(n, 0);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  return ((*rng)() % n);
}

INSTANTIATE_CLASS(DataTransformer);

}  // namespace caffe
