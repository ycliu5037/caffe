/*
TODO:
- only load parts of the file, in accordance with a prototxt param "max_mem"
*/

#include <stdint.h>
#include <vector>

#include "hdf5.h"
#include "hdf5_hl.h"

#include "caffe/layers/hdf5_data_layer2.hpp"

namespace caffe {

template <typename Dtype>
void HDF5Data2Layer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
// const int batch_size = this->layer_param_.hdf5_data_param2().batch_size();

  for (int n = 0; n < num_sources_; ++n) {
  
    for (int i = 0; i < batch_sizes_[n]; ++i, ++current_rows_[n]) {
      if (current_rows_[n] == hdf_blobs_[0]->shape(0)) {
        if (num_files_[n] > 1) {
          ++current_files_[n];
          if (current_files_[n] == num_files_[n]) {
            current_files_[n] = 0;
            if (this->layer_param_.hdf5_data_param2().shuffle()) {
              std::random_shuffle(file_permutations_[n].begin(),
                                  file_permutations_[n].end());
            }
            DLOG(INFO) << "Looping around to first file and reshuffling";
          }
          LoadHDF5FileData(
              hdf_filenames_[n][file_permutations_[n][ current_files_[n] ]].c_str());
        }
        current_rows_[n] = 0;
        if (this->layer_param_.hdf5_data_param2().shuffle())
          std::random_shuffle(data_permutation_.begin(), data_permutation_.end());
      }
      for (int j = 0; j < this->layer_param_.top_size(); ++j) {
        int data_dim = top[j]->count() / top[j]->shape(0);
        caffe_copy(data_dim,
            &hdf_blobs_[j]->cpu_data()[data_permutation_[current_rows_[n]]
              * data_dim], &top[j]->mutable_gpu_data()[i * data_dim]);
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(HDF5Data2Layer);

}  // namespace caffe
