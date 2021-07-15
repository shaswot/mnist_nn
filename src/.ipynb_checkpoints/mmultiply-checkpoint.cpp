#include <istream>
#include <iostream>
#include <fstream>

#include <stddef.h>
#include <typeinfo>
#include <stdexcept>

// https://github.com/arpaka/mnist-loader
#include "../include/mnist_loader.h" 

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xnpy.hpp>
#include <xtensor/xsort.hpp>

template <class _Tp>
xt::xarray<_Tp> softmax(xt::xarray<_Tp> a)
{
    xt::xarray<_Tp> temp = xt::exp(a);
    xt::xarray<_Tp> sum = xt::sum(temp);
    return temp/sum;
//     return sum;
//     return xt::reduce(function, input, axes)
}


template <class _Tp>
xt::xarray<_Tp> relu(xt::xarray<_Tp> a)
{
    return (xt::abs(a)+a)/2;
}
    

template <class _Tp>
xt::xarray<_Tp> matmul(xt::xarray<_Tp> a, 
                       xt::xarray<_Tp> b) noexcept(false)
{
//     if (a.shape().size() != b.shape().size()) {
//         throw std::runtime_error("Shape mismatching!");
//     }
    
    const unsigned int n = a.shape()[0]; // a rows
    const unsigned int m = a.shape()[1]; // a cols
    const unsigned int p = b.shape()[1]; // b cols

    xt::xarray<double>::shape_type shape = {n,p};
    xt::xarray<float> out = xt::zeros<_Tp>(shape);

    for (auto i = 0; i < n; ++i){
        for (auto j = 0; j < p; ++j){
            for (auto k = 0; k < m; ++k){
                out(i,j) += a(i,k) * b(k,j);
            }
        }
    }
    return out;
}


int main()
{
  
    // load weights from npy files
    xt::xarray<float> dense_weights = xt::load_npy<float>("../data/dense_weights.npy");
    xt::xarray<float> dense_biases = xt::load_npy<float>("../data/dense_biases.npy");
    dense_biases.reshape({-1, 1});
    
    xt::xarray<float> dense_1_weights = xt::load_npy<float>("../data/dense_1_weights.npy");
    xt::xarray<float> dense_1_biases = xt::load_npy<float>("../data/dense_1_biases.npy");
    dense_1_biases.reshape({-1, 1});

    // load mnist data
    mnist_loader train("../dataset/train-images-idx3-ubyte",
                     "../dataset/train-labels-idx1-ubyte", 60000);
    mnist_loader test("../dataset/t10k-images-idx3-ubyte",
                    "..//dataset/t10k-labels-idx1-ubyte", 10000);


    /*check for the image <image_no> and display truth label*/
    int image_no = 50365;
    int label = train.labels(image_no);
    std::cout << "IMAGE_NUMBER: " << image_no << std::endl;
    std::cout << "TRUTH_LABEL:  " << label << std::endl;
    
    // load the image <image_no> into vector and convert to xtensor<float32>
    std::vector<double> image = train.images(image_no);
        
    // cast to float32 from double and reshape to single batch size
    xt::xarray<float> input_image = xt::adapt(image);
    input_image.reshape({-1, 1});
    xt::dump_npy("../data/image_65.npy", input_image);

    
    // first layer
    // transpose weight matrix from (784,32) -> (32,784)
    auto tr_dense_weights = xt::transpose(dense_weights);    
    xt::xarray<float> l1 = matmul<float>(tr_dense_weights, input_image);
//     std::cout << "L1 SHAPE: " << xt::adapt(l1.shape()) << std::endl;
//     std::cout <<tr_dense_weights <<std::endl;
//     std::cout<<"***************" <<std::endl;
//     std::cout <<input_image <<std::endl;
//     std::cout<<"***************" <<std::endl;
//     std::cout <<l1 <<std::endl;
//     std::cout<<"***************" <<std::endl;
    
    // first layer bias
    xt::xarray<float> b1 = l1 + dense_biases;
//     std::cout <<b1 <<std::endl;
//     std::cout<<"***************" <<std::endl;
    
    // relu activation
    xt::xarray<float> b1_relu = relu<float>(b1);
//     std::cout <<b1_relu <<std::endl;
//     std::cout<<"***************" <<std::endl;
    
    // second layer
    auto tr_dense_1_weights = xt::transpose(dense_1_weights);
    
//     std::cout << "B1 SHAPE: " << xt::adapt(b1_relu.shape()) << std::endl;
//     std::cout << "tr dense SHAPE: " << xt::adapt(dense_1_weights.shape()) << std::endl;
    
    xt::xarray<float> l2 = matmul<float>(tr_dense_1_weights, b1_relu);
//     std::cout <<l2 <<std::endl;
//     std::cout<<"***************" <<std::endl;
    
    // second layer bias
    xt::xarray<float> b2 = l2 + dense_1_biases;
//     std::cout <<b2 <<std::endl;
//     std::cout<<"***************" <<std::endl;
    
    // softmax activation
    xt::xarray<float> l3 = softmax(b2);
//     std::cout <<l3 <<std::endl;
//     std::cout<<"***************" <<std::endl;
    
    // argmax
    std::cout << "PREDICTION:   " << xt::argmax(l3, 0)[0] << std::endl;
    
    return 0;
}

