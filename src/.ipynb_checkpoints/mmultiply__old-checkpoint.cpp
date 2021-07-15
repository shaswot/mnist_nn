#include <iostream>
#include <iomanip>
#include <vector>
#include "cnpy.h"
#include "mnist_loader.h"

// template <class T>
// std::vector <std::vector<T>> Multiply(std::vector <std::vector<T>> &a, 
//                                       std::vector <std::vector<T>> &b)
// {
//     const int n = a.size();     // a rows
//     const int m = a[0].size();  // a cols
//     const int p = b[0].size();  // b cols

//     std::vector <std::vector<T>> c(n, 
//                                    std::vector<T>(p, 0)); // initialized to zero
//     for (auto j = 0; j < p; ++j){
//         for (auto k = 0; k < m; ++k){
//             for (auto i = 0; i < n; ++i){
//                 c[i][j] += a[i][k] * b[k][j];
//             }
//         }
//     }
//     return c;
// }

// template <class T>
// int display_matrix(std::vector <std::vector<T>> &c){
//     for (auto i = 0; i <c.size(); ++i){
//         for (auto j = 0; j <c[i].size(); ++j){
//             std::cout<< std::setw(5) << c[i][j] << ' ';
//         }
//         std::cout << std::endl;
//     }
//     std::cout << std::endl;
//     return 0;
// }

// print a vector
// https://stackoverflow.com/a/31130991/7128154
template<typename T1>
std::ostream& operator <<( std::ostream& out, const std::vector<T1>& object )
{
      out << "[";
      if ( !object.empty() )
      {
          for(typename std::vector<T1>::const_iterator
              iter = object.begin();
              iter != --object.end();
              ++iter) {
                  out << *iter << ", ";
          }
          out << *--object.end();
      }
      out << "]"<<std::endl;
      return out;
}

int main(){
    
    //load it into a new array
    cnpy::NpyArray arr = cnpy::npy_load("./data/random_array.npy");
    float* loaded_data = arr.data<float>();
    
    std::cout<<"npy SHAPE: "<<arr.shape<<std::endl;
    std::cout<<"word SIZE: "<<arr.word_size<<" BYTES"<<std::endl;
    
    std::cout<<"npy DIMENSIONS: "<<arr.shape.size()<<std::endl;
    size_t npy_rows = arr.shape[0];
    size_t npy_cols = arr.shape[1];
    
    std::cout<<"no. of ROWS: "<<npy_rows<<std::endl;
    std::cout<<"no. of COLS: "<<npy_cols<<std::endl;
    
    // https://github.com/rogersce/cnpy/issues/20
    std::vector<std::vector<float>> vec2d;
    vec2d.reserve(npy_rows);
    for(size_t row = 0; row < npy_rows; row++) {
       vec2d.emplace_back(npy_cols);
       for(size_t col = 0; col < npy_cols; col++) {
            vec2d[row][col] = loaded_data[row*npy_cols + col];
       }
    }
    
    std::cout<<vec2d<<std::endl;
    
    

    
    
//     std::vector <std::vector<float>> npy_data = arr.data<std::vector <std::vector<float>>>();

    
//     float* loaded_data = arr.data<float>();
    
// //     std::vector<float> mydata(std::begin(loaded_data), std::end(loaded_data));
//     std::vector<float> mydata(loaded_data, loaded_data+arr.shape[0]);
    
//     for (std::vector<float>::const_iterator i = mydata.begin(); i != mydata.end(); ++i)
//     std::cout << *i << std::endl;

//     /*********************************************************************/

//     std::vector <std::vector<float> > a {{0,1,2,3},
//                                          {4,5,6,7},
//                                          {8,9,10,11}};

//     std::vector <std::vector<float> > b {{4,5,6,7,6},
//                                         {0,1,2,3,1},
//                                         {8,9,10,11,5},
//                                         {0,1,2,3,9}};

   

//     auto c = Multiply(a, b);
    
//     display_matrix(a);
//     display_matrix(b);
//     display_matrix(c);
    
//     /*********************************************************************/
    
//     //load it into a new array
//     cnpy::NpyArray arr = cnpy::npy_load("./data/dense_biases.npy");
//     std::cout<<"npy SHAPE: "<<arr.shape[0]<<std::endl;
//     std::cout<<"word SIZE: "<<arr.word_size<<" BYTES"<<std::endl;

//     float* loaded_data = arr.data<float>();
    
// //     std::vector<float> mydata(std::begin(loaded_data), std::end(loaded_data));
//     std::vector<float> mydata(loaded_data, loaded_data+arr.shape[0]);
    
//     for (std::vector<float>::const_iterator i = mydata.begin(); i != mydata.end(); ++i)
//     std::cout << *i << std::endl;
    
//     /*********************************************************************/

//     mnist_loader train("dataset/train-images-idx3-ubyte",
//                      "dataset/train-labels-idx1-ubyte", 100);
//     mnist_loader test("dataset/t10k-images-idx3-ubyte",
//                     "dataset/t10k-labels-idx1-ubyte", 100);

//     int rows  = train.rows();
//     int cols  = train.cols();

//     /*check for the image image_no*/

//     int image_no = 65;
//     int label = train.labels(image_no);
//     std::vector<double> image = train.images(image_no);

//     std::cout << "label: " << label << std::endl;
//     std::cout << "image: " << std::endl;
//     for (int y=0; y<rows; ++y) {
//     for (int x=0; x<cols; ++x) {
//       std::cout << ((image[y*cols+x] == 0.0)? ' ' : '*');
//     }
//     std::cout << std::endl;
//     }

//     std::cout<< "size of image array: "<<image.size()<<std::endl;


//     std::cout << std::fixed;
//     std::cout << std::setprecision(2);
//     std::cout<< "Printing the pixel values"<<image.size()<<std::endl;


//     for (int y=0; y<rows; ++y) {
//         for (int x=0; x<cols; ++x) {
//           std::cout <<image[y*cols+x]<< ' '; 
//         }
//     std::cout << std::endl;
//     }

//     /*********************************************************************/
//     https://stackoverflow.com/questions/28607912/sum-values-of-2-vectors/28607985
    
//     std::vector<int> a;//looks like this: 2,0,1,5,0
//     std::vector<int> b;//looks like this: 0,0,1,3,5

//     // std::plus adds together its two arguments:
//     std::transform (a.begin(), a.end(), b.begin(), a.begin(), std::plus<int>());
//     // a = 2,0,2,8,5
//     /*********************************************************************/    
    
    return 0;
}
