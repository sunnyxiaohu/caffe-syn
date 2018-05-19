#include "face-recog.hpp"

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0]
              << " deploy.prototxt network.caffemodel"
              << " img1.jpg img2.jpg" << std::endl;
        return 1;
    }
    
    ::google::InitGoogleLogging(argv[0]);

    string model_file   = argv[1];
    string trained_file = argv[2];
    FaceRecognition face_recog(model_file, trained_file);

    string img1_file    = argv[3];
    string img2_file   = argv[4];

    std::cout << "----- Computing for "
              << img1_file << " and "<< img2_file <<" -----" << std::endl;

    cv::Mat img1 = cv::imread(img1_file, -1);
    CHECK(!img1.empty()) << "Unable to decode image " << img1_file;
    cv::Mat img2 = cv::imread(img2_file, -1);
    CHECK(!img2.empty()) << "Unable to decode image " << img2_file;
  
    float similarity = face_recog.getSimilarity(img1, img2, true);

    /* Print the top N predictions. */
    std::cout << "Similarity: "<<similarity<<std::endl;

    return 0;
}
