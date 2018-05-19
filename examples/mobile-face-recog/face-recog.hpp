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
            return std::max<float>(0,getSimilarity(feat1,feat2));
        }
	
    private:
        shared_ptr<Net<float> > _net;
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
            for (int i = 0; i<result2.size(); ++i)
                result1.push_back(result2[i]);
            return result1;
	    }
	    /* gt the features of the last layer */
        vector<float> getLastLayerFeatures(const Mat& _img) {
            Mat img = _img.clone();
            img.convertTo(img, CV_32FC3);
            Blob<float>* inputBlob = _net->input_blobs()[0];
            int width = inputBlob->width();
            int height = inputBlob->height();
            resize(img, img, Size(width, height));
            // sub mean and rescaled to [-1, 1] according trainging prcedure
            // TODO: could be optimized by plane-copy (channel) operation
            img = (img - 127.5) * 0.0078125; 
            float* data = inputBlob->mutable_cpu_data();
            for (int k = 0; k<3; ++k){
                for (int i = 0; i<height; ++i){
                    for (int j = 0; j<width; ++j){
                        int index = (k*height + i)*width + j;
                        data[index] = img.at<Vec3f>(i, j)[k];
                    }
                }
            }
            vector<Blob<float>* > inputs(1, inputBlob);
            const vector<Blob<float>* >& outputBlobs = _net->Forward(inputs);
            Blob<float>* outputBlob = outputBlobs[0];
            const float* value = outputBlob->cpu_data();
            vector<float> result;
            for (int i = 0; i<outputBlob->count(); ++i)
	            result.push_back(value[i]);
            return result;
        }
};  

#endif // USE_OPENCV
#endif // FACE_RECOG_HPP
