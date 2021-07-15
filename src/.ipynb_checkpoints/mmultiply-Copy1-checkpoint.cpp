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

//     std::cout<<"n = "<<n<<std::endl;
//     std::cout<<"m = "<<m<<std::endl;
//     std::cout<<"p = "<<p<<std::endl;
    
    xt::xarray<double>::shape_type shape = {n,p};
    xt::xarray<float> out = xt::zeros<_Tp>(shape);

//     std::cout<<out<<std::endl;

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
  
//     // load weights from npy files
//     auto dense_weights = xt::load_npy<float>("../data/dense_weights.npy");
//     auto dense_biases = xt::load_npy<float>("../data/dense_biases.npy");
    
//     auto dense_1_weights = xt::load_npy<float>("../data/dense_1_weights.npy");
//     auto dense_1_biases = xt::load_npy<float>("../data/dense_1_biases.npy");

//     std::cout << dense_1_biases<<std::endl;
       
//     mnist_loader train("../dataset/train-images-idx3-ubyte",
//                      "../dataset/train-labels-idx1-ubyte", 100);
//     mnist_loader test("../dataset/t10k-images-idx3-ubyte",
//                     "..//dataset/t10k-labels-idx1-ubyte", 100);

// //     int rows  = train.rows();
// //     int cols  = train.cols();

//     /*check for the image <image_no> and display truth label*/
//     int image_no = 65;
//     int label = train.labels(image_no);
//     std::cout << "Loading train image no.: " << image_no << std::endl;
//     std::cout << "TRUTH label: " << label << std::endl;
    
//     // load the image into vector and convert to xtensor<float32>
//     std::vector<double> image = train.images(image_no);
//     auto input_image_double = xt::adapt(image);
//     // cast to float32 from double
//     auto input_image = xt::cast<float>(input_image_double);
// //     std::cout<<input_image[345]<<std::endl;
// //     std::cout<<"FLOAT DATA SIZE:"<<sizeof(input_image[345])<<" BYTES"<<std::endl;
    
    
    
//     //print shape of input_image
//     // https://github.com/xtensor-stack/xtensor/issues/1247
//     /* the shape of an xtensor is a regular std::array, 
//     and the shape of a xarray is a xt::svector which behaves like a std::vector*/
//     std::cout << "INPUT SHAPE: " << xt::adapt(input_image.shape()) << std::endl;
//     std::cout << "INPUT SHAPE SIZE: " << input_image.shape().size() << std::endl;

//     //print shape of dense_weight
//     std::cout << "DENSE SHAPE: " << xt::adapt(dense_weights.shape()) << std::endl;
//     std::cout << "DENSE SHAPE SIZE: " << dense_weights.shape().size() << std::endl;
    
//     std::cout << "dense_nrows: " << dense_weights.shape()[0] << std::endl;
//     std::cout << "dense_ncols: " << dense_weights.shape()[1] << std::endl;

    auto w_mat = xt::load_npy<float>("../data/random_array.npy");
    auto i_vec = xt::load_npy<float>("../data/random_input.npy");
    auto i_mat = xt::load_npy<float>("../data/random_input_mat.npy");

    i_vec.reshape({-1, 1});
    
    auto y1 = matmul<float>(w_mat, i_vec);
    auto y2 = matmul<float>(w_mat, i_mat);
    std::cout <<w_mat <<std::endl;
    std::cout <<i_vec <<std::endl;
    std::cout <<y1 <<std::endl;
    std::cout<<"***************" <<std::endl;
    
    std::cout <<w_mat <<std::endl;
    std::cout <<i_mat <<std::endl;
    std::cout <<y2 <<std::endl;

        


    
    return 0;
}

