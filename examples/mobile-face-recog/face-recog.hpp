#ifndef FACE_RECOG_HPP
#define FACE_RECOG_HPP

#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)
using namespace cv;
using std::string;
using std::vector;

class FaceRecognition {
    public:
        FaceRecognition(const string& model_file, const string& trained_file) {
            #ifdef CPU_ONLY
                Caffe::set_mode(Caffe::CPU);
            #else
                Caffe::set_mode(Caffe::GPU);
            #endif
            _net.reset(new Net<float>(model_file, TEST));
            _net->CopyTrainedLayersFrom(trained_file);
            
            CHECK_EQ(_net->num_inputs(), 1) << "Network should have exactly one input.";
            //CHECK_EQ(_net->num_outputs(), 1) << "Network should have exactly one output.";

            Blob<float>* input_layer = _net->input_blobs()[0];
            _num_channels = input_layer->channels();
            CHECK(_num_channels == 3 || _num_channels == 1)
            << "Input layer should have 1 or 3 channels.";
            _input_geometry = cv::Size(input_layer->width(), input_layer->height());
        }
     
        float getSimilarity(const Mat& lhs,const Mat& rhs,bool useFlipImg=true){
            vector<float> feat1,feat2;
            if(useFlipImg){
                feat1=getLastLayerFeaturesFlip(lhs);
                feat2=getLastLayerFeaturesFlip(rhs);
            }
            else{
                feat1=getLastLayerFeatures(lhs);
                feat2=getLastLayerFeatures(rhs);
            }
            return 0;
            //return std::max<float>(0,getSimilarity(feat1,feat2));
        }
	
    private:
        shared_ptr<Net<float> > _net;
        cv::Size _input_geometry;
        int _num_channels;
        
        /* get moldel */
        float getMold(const vector<float>& vec) {
            int n = vec.size();
            float sum = 0.0;
            for (int i = 0; i<n; ++i)
                sum += vec[i] * vec[i];
            return sqrt(sum);
	    }
	    /* get the cosine similarity */
	    float getSimilarity(const vector<float>& lhs, const vector<float>& rhs) {
            int n = lhs.size();
            assert(n == rhs.size());
            float tmp = 0.0; // inner product
            for (int i = 0; i<n; ++i)
                tmp += lhs[i] * rhs[i];
            return tmp / (getMold(lhs)*getMold(rhs));
	    }
	    /* get the flipped features of the last layer */
	    vector<float> getLastLayerFeaturesFlip(const Mat& _img) {
            vector<float> result1 = getLastLayerFeatures(_img);
            Mat flipImg;
            flip(_img, flipImg, 0);    // top-down flip
            vector<float> result2 = getLastLayerFeatures(flipImg);
            //for (int i = 0; i<result2.size(); ++i)
            //    result1.push_back(result2[i]);
            return result1;
	    }
	    /* gt the features of the last layer */
        vector<float> getLastLayerFeatures(const Mat& _img) {
            Mat img = _img.clone();
            
            Blob<float>* input_layer = _net->input_blobs()[0];
            input_layer->Reshape(1, _num_channels, _input_geometry.height, _input_geometry.width);
            // Forward dimension change to all layers.
            _net->Reshape();

            std::vector<cv::Mat> input_channels;
            wrapInputLayer(&input_channels);
            preprocess(img, &input_channels);
            const vector<Blob<float>* >& output_layers = _net->Forward();
            
            // copy the output layer to a std::vector
            Blob<float>* output_layer = output_layers[0];
            const float* begin = output_layer->cpu_data();
            const float* end = begin + output_layer->count();
            return std::vector<float>(begin, end);
        }
        /* Wrap the input layer of the network in separate cv::Mat objects
        * (one per channel). This way we save one memcpy operation and we
        * don't need to rely on cudaMemcpy2D. The last preprocessing
        * operation will write the separacte channels directly to the input
        * layer. 
        */
        void wrapInputLayer(vector<Mat>* input_channels) {
            Blob<float>* input_layer = _net->input_blobs()[0];
            int width = input_layer->width();
            int height = input_layer->height();
            float* input_data = input_layer->mutable_cpu_data();
            for (int i = 0; i < input_layer->channels(); ++i) {
                cv::Mat channel(height, width, CV_32FC1, input_data);
                input_channels->push_back(channel);
                input_data += width * height;
            }
        }
        void preprocess(const Mat& img, std::vector<cv::Mat>* input_channels) {
            /* Convert the input image to the input image format of the network. */
            cv::Mat sample;
            if (img.channels() == 3 && _num_channels == 1)
            cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
            else if (img.channels() == 4 && _num_channels == 1)
            cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
            else if (img.channels() == 4 && _num_channels == 3)
            cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
            else if (img.channels() == 1 && _num_channels == 3)
            cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
            else
            sample = img;
            
            cv::Mat sample_resized;
            if (sample.size() != _input_geometry)
                cv::resize(sample, sample_resized, _input_geometry);
            else
                sample_resized = sample;

            cv::Mat sample_float;
            if (_num_channels == 3)
                sample_resized.convertTo(sample_float, CV_32FC3);
            else
                sample_resized.convertTo(sample_float, CV_32FC1);

            cv::Mat sample_normalized;
            // sub mean and rescaled to [-1, 1] according trainging prcedure
            sample_normalized = (sample_float - 127.5) * 0.0078125;
            /* This operation will write the separate BGR planes directly to the
            * input layer of the network because it is wrapped by the cv::Mat
            * objects in input_channels. */
            cv::split(sample_normalized, *input_channels);
        }
};  

#endif // USE_OPENCV
#endif // FACE_RECOG_HPP
