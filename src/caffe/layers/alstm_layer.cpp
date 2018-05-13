#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/alstm_layer.hpp"
#include "caffe/layers/lstm_layer.hpp"
#include "caffe/util/math_functions.hpp"

//Todo: parameters sharing among different ALSTMs
namespace caffe {

template <typename Dtype>
void ALSTMLayer<Dtype>::RecurrentInputBlobNames(vector<string>* names) const {
  names->resize(2);
  (*names)[0] = "h_0";
  (*names)[1] = "c_0";
}

template <typename Dtype>
void ALSTMLayer<Dtype>::RecurrentOutputBlobNames(vector<string>* names) const {
  names->resize(2);
  (*names)[0] = "h_" + format_int(this->T_);
  (*names)[1] = "c_T";
}

template <typename Dtype>
void ALSTMLayer<Dtype>::RecurrentInputShapes(vector<BlobShape>* shapes) const {
  const int num_output = this->layer_param_.recurrent_param().num_output();
  const int num_blobs = 2;
  shapes->resize(num_blobs);
  for (int i = 0; i < num_blobs; ++i) {
    (*shapes)[i].Clear();
    (*shapes)[i].add_dim(1);  // a single timestep
    (*shapes)[i].add_dim(this->N_);
    (*shapes)[i].add_dim(num_output);
  }
}

template <typename Dtype>
void ALSTMLayer<Dtype>::OutputBlobNames(vector<string>* names) const {
  names->resize(2);
  (*names)[0] = "h";
  (*names)[1] = "mask";
}

template <typename Dtype>
void ALSTMLayer<Dtype>::FillUnrolledNet(NetParameter* net_param) const {
  const int num_output = this->layer_param_.recurrent_param().num_output();
  CHECK_GT(num_output, 0) << "num_output must be positive";
  const FillerParameter& weight_filler =
      this->layer_param_.recurrent_param().weight_filler();
  const FillerParameter& bias_filler =
      this->layer_param_.recurrent_param().bias_filler();

  // Take bottom[1].shape=(9,80,384,36) and recurrent_param().num_output()=256 for illustrating, 
  // where N=80, T=9, C=384, K^2=36

  // Add generic LayerParameter's (without bottoms/tops) of layer types we'll
  // use to save redundant code.
  LayerParameter hidden_param;
  hidden_param.set_type("InnerProduct");
  hidden_param.mutable_inner_product_param()->set_num_output(num_output * 4);
  hidden_param.mutable_inner_product_param()->set_bias_term(false);
  hidden_param.mutable_inner_product_param()->set_axis(2);
  hidden_param.mutable_inner_product_param()->
      mutable_weight_filler()->CopyFrom(weight_filler);

  LayerParameter biased_hidden_param(hidden_param);
  biased_hidden_param.mutable_inner_product_param()->set_bias_term(true);
  biased_hidden_param.mutable_inner_product_param()->
      mutable_bias_filler()->CopyFrom(bias_filler);

  LayerParameter attention_param(hidden_param);  
  attention_param.mutable_inner_product_param()->set_num_output(num_output);  
  
  LayerParameter biased_attention_param(attention_param);  
  biased_attention_param.mutable_inner_product_param()->set_bias_term(true);  
  biased_attention_param.mutable_inner_product_param()->  
      mutable_bias_filler()->CopyFrom(bias_filler); // 

  LayerParameter sum_param;
  sum_param.set_type("Eltwise");
  sum_param.mutable_eltwise_param()->set_operation(
      EltwiseParameter_EltwiseOp_SUM);

  LayerParameter scale_param;
  scale_param.set_type("Scale");
  scale_param.mutable_scale_param()->set_axis(0);

  LayerParameter slice_param;
  slice_param.set_type("Slice");
  slice_param.mutable_slice_param()->set_axis(0);

  LayerParameter softmax_param;  
  softmax_param.set_type("Softmax");  
  softmax_param.mutable_softmax_param()->set_axis(1);

  LayerParameter split_param;
  split_param.set_type("Split");

  LayerParameter permute_param;
  permute_param.set_type("Permute");

  LayerParameter reshape_param;
  reshape_param.set_type("Reshape");

  LayerParameter bias_param;  
  bias_param.set_type("Bias");

  LayerParameter pool_param;
  pool_param.set_type("Pooling");

  vector<BlobShape> input_shapes;
  RecurrentInputShapes(&input_shapes);
  CHECK_EQ(2, input_shapes.size());

  LayerParameter* input_layer_param = net_param->add_layer();
  input_layer_param->set_type("Input");
  InputParameter* input_param = input_layer_param->mutable_input_param();

  input_layer_param->add_top("c_0");
  input_param->add_shape()->CopyFrom(input_shapes[0]);

  input_layer_param->add_top("h_0");
  input_param->add_shape()->CopyFrom(input_shapes[1]);

  LayerParameter* cont_slice_param = net_param->add_layer();
  cont_slice_param->CopyFrom(slice_param);
  cont_slice_param->set_name("cont_slice");
  cont_slice_param->add_bottom("cont");
  cont_slice_param->mutable_slice_param()->set_axis(0);

  LayerParameter* x_slice_param = net_param->add_layer();
  x_slice_param->CopyFrom(slice_param);
  x_slice_param->add_bottom("x");
  x_slice_param->set_name("x_slice");

  // Add layer to transform all timesteps of x to the hidden state dimension.
  //     W_xc_x = W_xc * x + b_c
/*
  {
    LayerParameter* x_transform_param = net_param->add_layer();
    x_transform_param->CopyFrom(biased_hidden_param);
    x_transform_param->set_name("x_transform");
    x_transform_param->add_param()->set_name("W_xc");
    x_transform_param->add_param()->set_name("b_c");
    x_transform_param->add_bottom("x");
    x_transform_param->add_top("W_xc_x");
    x_transform_param->add_propagate_down(true);
  }
*/
  if (this->static_input_) {
    // Add layer to transform x_static to the gate dimension.
    //     W_xc_x_static = W_xc_static * x_static
    LayerParameter* x_static_transform_param = net_param->add_layer();
    x_static_transform_param->CopyFrom(hidden_param);
    x_static_transform_param->mutable_inner_product_param()->set_axis(1);
    x_static_transform_param->set_name("W_xc_x_static");
    x_static_transform_param->add_param()->set_name("W_xc_static");
    x_static_transform_param->add_bottom("x_static");
    x_static_transform_param->add_top("W_xc_x_static_preshape");
    x_static_transform_param->add_propagate_down(true);

    LayerParameter* reshape_static_param = net_param->add_layer();
    reshape_static_param->CopyFrom(reshape_param);
    BlobShape* new_shape =
         reshape_static_param->mutable_reshape_param()->mutable_shape();
    new_shape->add_dim(1);  // One timestep.
    // Should infer this->N as the dimension so we can reshape on batch size.
    new_shape->add_dim(-1);
    new_shape->add_dim(
        x_static_transform_param->inner_product_param().num_output());
    reshape_static_param->set_name("W_xc_x_static_reshape");
    reshape_static_param->add_bottom("W_xc_x_static_preshape");
    reshape_static_param->add_top("W_xc_x_static");
  }
/*
  LayerParameter* x_slice_param = net_param->add_layer();
  x_slice_param->CopyFrom(slice_param);
  x_slice_param->add_bottom("W_xc_x");
  x_slice_param->set_name("W_xc_x_slice");
*/
  LayerParameter output_concat_layer;
  output_concat_layer.set_name("h_concat");
  output_concat_layer.set_type("Concat");
  output_concat_layer.add_top("h");
  output_concat_layer.mutable_concat_param()->set_axis(0);

  LayerParameter output_m_layer;  
  output_m_layer.set_name("m_concat");  
  output_m_layer.set_type("Concat");  
  output_m_layer.add_top("mask");  
  output_m_layer.mutable_concat_param()->set_axis(0); // output attention mask

  for (int t = 1; t <= this->T_; ++t) {
    string tm1s = format_int(t - 1);
    string ts = format_int(t);

    cont_slice_param->add_top("cont_" + ts);
    x_slice_param->add_top("x_" + ts);//(1,80,384,36)

    // Add a layer to permute x: for x_data 
    {  
      LayerParameter* permute_x_param = net_param->add_layer();  
      permute_x_param->CopyFrom(permute_param);  
      permute_x_param->set_name("permute_x_" + ts);  
      permute_x_param->mutable_permute_param()->add_order(2);  
      permute_x_param->mutable_permute_param()->add_order(0);  
      permute_x_param->mutable_permute_param()->add_order(1);  
      permute_x_param->mutable_permute_param()->add_order(3);  
      permute_x_param->add_bottom("x_" + ts);  
      permute_x_param->add_top("x_p_" + ts);  //(384,1,80,36)
    }  

    // Add a layer to generate attention weights  
    {  
      LayerParameter* att_m_param = net_param->add_layer();  
      att_m_param->CopyFrom(attention_param);  
      att_m_param->set_name("att_m_" + tm1s);  
      att_m_param->add_bottom("h_" + tm1s); //(1,80,256) 
      att_m_param->add_top("m_" + tm1s);     //(1,80,256)
    }  

    {  
      LayerParameter* permute_x_a_param = net_param->add_layer();  
      permute_x_a_param->CopyFrom(permute_param);  
      permute_x_a_param->set_name("permute_x_a_" + ts);  
      permute_x_a_param->mutable_permute_param()->add_order(3);  
      permute_x_a_param->mutable_permute_param()->add_order(0);  
      permute_x_a_param->mutable_permute_param()->add_order(1);  
      permute_x_a_param->mutable_permute_param()->add_order(2);  
      permute_x_a_param->add_bottom("x_" + ts);  
      permute_x_a_param->add_top("x_p_a_" + ts);  //(36,1,80,384)
    }   

    {  
      LayerParameter* att_x_param = net_param->add_layer();  
      att_x_param->CopyFrom(attention_param);  
      att_x_param->set_name("att_x_" + tm1s);  
      att_x_param->mutable_inner_product_param()->set_axis(3);  
      att_x_param->add_bottom("x_p_a_" + ts);  
      att_x_param->add_top("m_x_a_" + tm1s);  //(36,1,80,256)
    } 
    
    // m_input := 
    //         := m_x_a_{t-1} + m_{t-1}
    {  
      LayerParameter* m_sum_layer = net_param->add_layer();  
      m_sum_layer->CopyFrom(bias_param);  
      m_sum_layer->set_name("mask_input_" + ts);  
      m_sum_layer->add_bottom("m_x_a_" + tm1s);  
      m_sum_layer->add_bottom("m_" + tm1s);  
      m_sum_layer->add_top("m_input_" + tm1s);  //(36,1,80,256)
    }  

    {  
      LayerParameter* att_x_ap_param = net_param->add_layer();  
      att_x_ap_param->CopyFrom(biased_attention_param);  
      att_x_ap_param->set_name("att_x_ap_" + tm1s);  
      att_x_ap_param->mutable_inner_product_param()->set_axis(3);  
      att_x_ap_param->mutable_inner_product_param()->set_num_output(1);  
      att_x_ap_param->add_bottom("m_input_" + tm1s);  
      att_x_ap_param->add_top("m_x_ap_" + tm1s);  //(36,1,80,1)
    } 
 
    {  
      LayerParameter* permute_m_param = net_param->add_layer();  
      permute_m_param->CopyFrom(permute_param);  
      permute_m_param->set_name("permute_m_" + ts);  
      permute_m_param->mutable_permute_param()->add_order(1);  
      permute_m_param->mutable_permute_param()->add_order(2);  
      permute_m_param->mutable_permute_param()->add_order(0);  
      permute_m_param->mutable_permute_param()->add_order(3);  
      permute_m_param->add_bottom("m_x_ap_" + tm1s);  
      permute_m_param->add_top("m_f_" + tm1s);  //(1,80,36,1)
    }  

    // Add a softmax layers to generate attention masks  
    {  
      LayerParameter* softmax_m_param = net_param->add_layer();  
      softmax_m_param->CopyFrom(softmax_param);  
      softmax_m_param->mutable_softmax_param()->set_axis(2);  
      softmax_m_param->set_name("softmax_m_" + tm1s);  
      softmax_m_param->add_bottom("m_f_" + tm1s);  
      softmax_m_param->add_top("mask_" + tm1s); //(1,80,36,1)
    }  

    {  
      LayerParameter* reshape_m_param = net_param->add_layer();  
      reshape_m_param->CopyFrom(reshape_param);  
      BlobShape* new_shape = reshape_m_param->mutable_reshape_param()->mutable_shape();  
      new_shape->Clear();  
      new_shape->add_dim(0);  
      new_shape->add_dim(0);  
      new_shape->add_dim(0);  
      reshape_m_param->set_name("reshape_m_" + tm1s);  
      reshape_m_param->add_bottom("mask_" + tm1s);  
      reshape_m_param->add_top("mask_reshape_" + tm1s); //(1,80,36) 
    } 

    // Conbine mask with input features  
    {  
      LayerParameter* scale_x_param = net_param->add_layer();  
      scale_x_param->CopyFrom(scale_param);  
      scale_x_param->mutable_scale_param()->set_axis(1);
      scale_x_param->set_name("scale_x_" + tm1s);  
      scale_x_param->add_bottom("x_p_" + ts);  
      scale_x_param->add_bottom("mask_reshape_" + tm1s);  
      scale_x_param->add_top("x_mask_" + ts);  //(384,1,80,36)
    }  

    {  
      LayerParameter* pool_x_param = net_param->add_layer();  
      pool_x_param->CopyFrom(pool_param);  
      pool_x_param->set_name("pool_x_"+ts);  
      pool_x_param->mutable_pooling_param()->set_pool(PoolingParameter_PoolMethod_AVE);  //Todo: PoolingParameter_PoolMethod_SUM
      pool_x_param->mutable_pooling_param()->set_kernel_h(1);//H=80, W=36 
      pool_x_param->mutable_pooling_param()->set_kernel_w(this->Loc_);
      pool_x_param->add_bottom("x_mask_"+ts);  
      pool_x_param->add_top("x_pool_"+ts);  //(384,1,80)
    }  

    {  
      LayerParameter* permute_x_pool_param = net_param->add_layer();  
      permute_x_pool_param->CopyFrom(permute_param);  
      permute_x_pool_param->set_name("permute_x_pool_" + ts);  
      permute_x_pool_param->mutable_permute_param()->add_order(1);  
      permute_x_pool_param->mutable_permute_param()->add_order(2); 
      permute_x_pool_param->mutable_permute_param()->add_order(0);   
      permute_x_pool_param->add_bottom("x_pool_" + ts);  
      permute_x_pool_param->add_top("x_pool_permute_" + ts); //(1,80,384)  
    }  

    // Add layer to transform a timestep of x_pool_permute to the hidden state dimension.
    //     W_xc_x_ = W_xc * x_pool_permute + b_c
    {  
      LayerParameter* x_transform_param = net_param->add_layer();  
      x_transform_param->CopyFrom(biased_hidden_param);  
      x_transform_param->set_name("x_transform_" + ts);  
      x_transform_param->add_param()->set_name("W_xc_" + ts);  
      x_transform_param->add_param()->set_name("b_c" + ts);  
      x_transform_param->add_bottom("x_pool_permute_" + ts);  
      x_transform_param->add_top("W_xc_x_" + ts);  //(1,80,1024)
    }  

    // Add layers to flush the hidden state when beginning a new
    // sequence, as indicated by cont_t.
    //     h_conted_{t-1} := cont_t * h_{t-1}
    //
    // Normally, cont_t is binary (i.e., 0 or 1), so:
    //     h_conted_{t-1} := h_{t-1} if cont_t == 1
    //                       0   otherwise
    {
      LayerParameter* cont_h_param = net_param->add_layer();
      cont_h_param->CopyFrom(scale_param);
      cont_h_param->set_name("h_conted_" + tm1s);
      cont_h_param->add_bottom("h_" + tm1s);
      cont_h_param->add_bottom("cont_" + ts);
      cont_h_param->add_top("h_conted_" + tm1s); //(1,80,256)
    }

    // Add layer to compute
    //     W_hc_h_{t-1} := W_hc * h_conted_{t-1}
    {
      LayerParameter* w_param = net_param->add_layer();
      w_param->CopyFrom(hidden_param);
      w_param->set_name("transform_" + ts);
      w_param->add_param()->set_name("W_hc");
      w_param->add_bottom("h_conted_" + tm1s);
      w_param->add_top("W_hc_h_" + tm1s); //(1,80,1024)
      w_param->mutable_inner_product_param()->set_axis(2); 
    }

    // Add the outputs of the linear transformations to compute the gate input.
    //     gate_input_t := W_hc * h_conted_{t-1} + W_xc * x_t + b_c
    //                   = W_hc_h_{t-1} + W_xc_x_t + b_c
    {
      LayerParameter* input_sum_layer = net_param->add_layer();
      input_sum_layer->CopyFrom(sum_param);
      input_sum_layer->set_name("gate_input_" + ts);
      input_sum_layer->add_bottom("W_hc_h_" + tm1s);
      input_sum_layer->add_bottom("W_xc_x_" + ts);
      if (this->static_input_) {
        input_sum_layer->add_bottom("W_xc_x_static"); //(1,80,1024)
      }
      input_sum_layer->add_top("gate_input_" + ts); //(1,80,1024)
    }

    // Add LSTMUnit layer to compute the cell & hidden vectors c_t and h_t.
    // Inputs: c_{t-1}, gate_input_t = (i_t, f_t, o_t, g_t), cont_t
    // Outputs: c_t, h_t
    //     [ i_t' ]
    //     [ f_t' ] := gate_input_t
    //     [ o_t' ]
    //     [ g_t' ]
    //         i_t := \sigmoid[i_t']
    //         f_t := \sigmoid[f_t']
    //         o_t := \sigmoid[o_t']
    //         g_t := \tanh[g_t']
    //         c_t := cont_t * (f_t .* c_{t-1}) + (i_t .* g_t)
    //         h_t := o_t .* \tanh[c_t]
    {
      LayerParameter* lstm_unit_param = net_param->add_layer();
      lstm_unit_param->set_type("LSTMUnit");
      lstm_unit_param->add_bottom("c_" + tm1s); //(1,80,256)
      lstm_unit_param->add_bottom("gate_input_" + ts);//(1,80,1024)
      lstm_unit_param->add_bottom("cont_" + ts);//(1,80,256)
      lstm_unit_param->add_top("c_" + ts);//(1,80,256)
      lstm_unit_param->add_top("h_" + ts);//(1,80,256)
      lstm_unit_param->set_name("unit_" + ts);
    }
    output_concat_layer.add_bottom("h_" + ts);
    output_m_layer.add_bottom("mask_reshape_" + tm1s);
  }  // for (int t = 1; t <= this->T_; ++t)

  {
    LayerParameter* c_T_copy_param = net_param->add_layer();
    c_T_copy_param->CopyFrom(split_param);
    c_T_copy_param->add_bottom("c_" + format_int(this->T_));
    c_T_copy_param->add_top("c_T");
  }
  net_param->add_layer()->CopyFrom(output_concat_layer);
  net_param->add_layer()->CopyFrom(output_m_layer);
}

INSTANTIATE_CLASS(ALSTMLayer);
REGISTER_LAYER_CLASS(ALSTM);

}  // namespace caffe
