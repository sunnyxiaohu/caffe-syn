// ------------------------------------------------------------------
// R-FCN
// Copyright (c) 2016 Microsoft
// Licensed under The MIT License [see r-fcn/LICENSE for details]
// Written by Yi Li
// ------------------------------------------------------------------

#include <cfloat>

#include <string>
#include <utility>
#include <vector>

#include "caffe/layers/rfcn_layers.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {
  template <typename Dtype>
  void PSROIPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top){
    PSROIPoolingParameter psroi_pooling_param = this->layer_param_.psroi_pooling_param();
    spatial_scale_ = psroi_pooling_param.spatial_scale();
    LOG(INFO) << "Spatial scale: " << spatial_scale_;

    CHECK_GT(psroi_pooling_param.output_dim(), 0)
      << "output_dim must be > 0";
    CHECK_GT(psroi_pooling_param.group_size(), 0)
      << "group_size must be > 0";

    output_dim_ = psroi_pooling_param.output_dim();
    group_size_ = psroi_pooling_param.group_size();
    pooled_height_ = group_size_;
    pooled_width_ = group_size_;
  }

  template <typename Dtype>
  void PSROIPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    channels_ = bottom[0]->channels();
    CHECK_EQ(channels_, output_dim_*group_size_*group_size_)
      << "input channel number does not match layer parameters";
    height_ = bottom[0]->height();
    width_ = bottom[0]->width();
    top[0]->Reshape(bottom[1]->num(), output_dim_, pooled_height_, pooled_width_);
    mapping_channel_.Reshape(bottom[1]->num(), output_dim_, pooled_height_, pooled_width_);
  }

  template <typename Dtype>
  void PSROIPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top){
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_rois = bottom[1]->cpu_data();
  int* mapping_channel = mapping_channel_.mutable_cpu_data();
  // Number of ROIs
  int num_rois = bottom[1]->num();
  int batch_size = bottom[0]->num();
  int top_count = top[0]->count();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top_count, Dtype(-FLT_MAX), top_data);
  // int* argmax_data = max_idx_.mutable_cpu_data();
  // caffe_set(top_count, -1, argmax_data);

  // For each ROI R = [batch_index x1 y1 x2 y2]:
  for (int n = 0; n < num_rois; ++n) {
    int roi_batch_ind = bottom_rois[0];
    Dtype roi_start_w = static_cast<Dtype>(round(bottom_rois[1])) * spatial_scale_;
    Dtype roi_start_h = static_cast<Dtype>(round(bottom_rois[2])) * spatial_scale_;
    Dtype roi_end_w = static_cast<Dtype>(round(bottom_rois[3]) + 1.) * spatial_scale_;
    Dtype roi_end_h = static_cast<Dtype>(round(bottom_rois[4]) + 1.) * spatial_scale_;
    CHECK_GE(roi_batch_ind, 0);
    CHECK_LT(roi_batch_ind, batch_size);

    Dtype roi_height = max(roi_end_h - roi_start_h, static_cast<Dtype>(0.1)); //avoid 0
    Dtype roi_width = max(roi_end_w - roi_start_w, static_cast<Dtype>(0.1));
    const Dtype bin_size_h = static_cast<Dtype>(roi_height)
                             / static_cast<Dtype>(pooled_height_);
    const Dtype bin_size_w = static_cast<Dtype>(roi_width)
                             / static_cast<Dtype>(pooled_width_);

    const Dtype* batch_data; // = bottom_data; // + bottom[0]->offset(roi_batch_ind);

    for (int ctop = 0; ctop < output_dim_; ++ctop) {
      for (int ph = 0; ph < pooled_height_; ++ph) {
        for (int pw = 0; pw < pooled_width_; ++pw) {
          // Compute pooling region for this output unit:
          //  start (included) = floor(ph * roi_height / pooled_height_)
          //  end (excluded) = ceil((ph + 1) * roi_height / pooled_height_)
          int hstart = static_cast<int>(floor(static_cast<Dtype>(ph)
                                              * bin_size_h));
          int wstart = static_cast<int>(floor(static_cast<Dtype>(pw)
                                              * bin_size_w));
          int hend = static_cast<int>(ceil(static_cast<Dtype>(ph + 1)
                                           * bin_size_h));
          int wend = static_cast<int>(ceil(static_cast<Dtype>(pw + 1)
                                           * bin_size_w));

          hstart = min(max(hstart + roi_start_h, static_cast<Dtype>(0)), static_cast<Dtype>(height_));
          hend = min(max(hend + roi_start_h, static_cast<Dtype>(0)), static_cast<Dtype>(height_));
          wstart = min(max(wstart + roi_start_w, static_cast<Dtype>(0)), static_cast<Dtype>(width_));
          wend = min(max(wend + roi_start_w, static_cast<Dtype>(0)), static_cast<Dtype>(width_));

          bool is_empty = (hend <= hstart) || (wend <= wstart);

          int gw = pw;
          int gh = ph;
          int c = (ctop*group_size_ + gh)*group_size_ + gw; // map to the channels of bottom_data
          // LOG(INFO)<<"ctop: "<<ctop<<", c: "<<c;
          batch_data = bottom_data + (roi_batch_ind * channels_ + c) * height_ * width_;
          // batch_data += c * height_ * width_;
          Dtype out_sum = 0;
          for (int h = hstart; h < hend; ++h){
            for (int w = wstart; w < wend; ++w){
              int bottom_index = h*width_ + w;
              out_sum += batch_data[bottom_index];
            }
          }

          Dtype bin_area = (hend - hstart)*(wend - wstart);
          const int pool_index = ph * pooled_width_ + pw;
          top_data[pool_index] = is_empty? 0. : out_sum/bin_area;
          mapping_channel[pool_index] = c;
        }
      }
      // Increment all data pointers by one channel
      // batch_data += bottom[0]->offset(0, group_size_*group_size_);
      top_data += top[0]->offset(0, 1);
      mapping_channel += mapping_channel_.offset(0, 1);
    } 
    // Increment ROI data pointer
    bottom_rois += bottom[1]->offset(1);
  }
    // NOT_IMPLEMENTED;
  }

  template <typename Dtype>
  void PSROIPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
    NOT_IMPLEMENTED;
  }
#ifdef CPU_ONLY
  STUB_GPU(PSROIPoolingLayer);
#endif

  INSTANTIATE_CLASS(PSROIPoolingLayer);
  REGISTER_LAYER_CLASS(PSROIPooling);

}
