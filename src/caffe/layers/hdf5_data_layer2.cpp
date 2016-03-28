/*
TODO:
- load file in a separate thread ("prefetch")
- can be smarter about the memcpy call instead of doing it row-by-row
  :: use util functions caffe_copy, and Blob->offset()
  :: don't forget to update hdf5_daa_layer.cu accordingly
- add ability to shuffle filenames if flag is set
*/
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>

#include "hdf5.h"
#include "hdf5_hl.h"
#include "stdint.h"

#include "caffe/layers/hdf5_data_layer2.hpp"
#include "caffe/util/hdf5.hpp"

namespace caffe {

template <typename Dtype>
HDF5Data2Layer<Dtype>::~HDF5Data2Layer<Dtype>() { }

// Load data and label from HDF5 filename into the class property blobs.
template <typename Dtype>
void HDF5Data2Layer<Dtype>::LoadHDF5FileData(const char* filename) {
  DLOG(INFO) << "Loading HDF5 file: " << filename;
  hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id < 0) {
    LOG(FATAL) << "Failed opening HDF5 file: " << filename;
  }

  int top_size = this->layer_param_.top_size();
  hdf_blobs_.resize(top_size);

  const int MIN_DATA_DIM = 1;
  const int MAX_DATA_DIM = INT_MAX;

  for (int i = 0; i < top_size; ++i) {
    hdf_blobs_[i] = shared_ptr<Blob<Dtype> >(new Blob<Dtype>());
    hdf5_load_nd_dataset(file_id, this->layer_param_.top(i).c_str(),
        MIN_DATA_DIM, MAX_DATA_DIM, hdf_blobs_[i].get());
  }

  herr_t status = H5Fclose(file_id);
  CHECK_GE(status, 0) << "Failed to close HDF5 file: " << filename;

  // MinTopBlobs==1 guarantees at least one top blob
  CHECK_GE(hdf_blobs_[0]->num_axes(), 1) << "Input must have at least 1 axis.";
  const int num = hdf_blobs_[0]->shape(0);
  for (int i = 1; i < top_size; ++i) {
    CHECK_EQ(hdf_blobs_[i]->shape(0), num);
  }

  // Default to identity permutation.
  data_permutation_.clear();
  data_permutation_.resize(hdf_blobs_[0]->shape(0));
  for (int i = 0; i < hdf_blobs_[0]->shape(0); i++)
    data_permutation_[i] = i;

  // Shuffle if needed.
  if (this->layer_param_.hdf5_data_param2().shuffle()==2) {
    std::random_shuffle(data_permutation_.begin(), data_permutation_.end());
    DLOG(INFO) << "Successully loaded " << hdf_blobs_[0]->shape(0)
               << " rows (shuffled)";
  } else {
    DLOG(INFO) << "Successully loaded " << hdf_blobs_[0]->shape(0) << " rows";
  }
}

template <typename Dtype>
void HDF5Data2Layer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Refuse transformation parameters since HDF5 is totally generic.
  CHECK(!this->layer_param_.has_transform_param()) <<
      this->type() << " does not transform data.";

  

  const int num_batch_size = this->layer_param_.hdf5_data_param2().batch_size_size();
  num_sources_ = this->layer_param_.hdf5_data_param2().source_size();

  CHECK(num_batch_size==num_sources_||num_batch_size==1)
      << "the batch size must be specificed once or once per source.";

  LOG(INFO) << "Number of sources files specified: " << num_sources_;
  
  batch_size_ = 0;
  batch_sizes_.clear();
  ostringstream stream;
  for (int n = 0; n < num_sources_; ++n) {
    batch_size_+=this->layer_param_.hdf5_data_param2().batch_size((num_batch_size==1) ? 0 : n);
    batch_sizes_.push_back(this->layer_param_.hdf5_data_param2().batch_size((num_batch_size==1) ? 0 : n));
    stream << batch_sizes_[n] << " ";
  }
  LOG(INFO) << "Batch Sizes: ( " << stream.str() << ") = " <<batch_size_;


  
  // Read the source to parse the filenames.
  
  std::vector<unsigned int> file_permutation;
  std::vector<unsigned int> data_permutation;
  hdf_filenames_.resize(num_sources_);
  current_rows_.clear();
  data_permutation_.clear();
  file_permutations_.clear();
  for (int n = 0; n < num_sources_; ++n) {

    const string& source = this->layer_param_.hdf5_data_param2().source(n);
    LOG(INFO) << "Loading list of HDF5 filenames from: " << source;
    
    std::ifstream source_file(source.c_str());
    if (source_file.is_open()) {
      std::string line;
      while (source_file >> line) {
        hdf_filenames_[n].push_back(line);
      }
    } else {
      LOG(FATAL) << "Failed to open source file: " << source;
    }
    source_file.close();
    num_files_.push_back(hdf_filenames_[n].size());
    current_files_.push_back(0);
    current_rows_.push_back(0);
    LOG(INFO) << "Number of HDF5 files: " << num_files_[n];
    CHECK_GE(num_files_[n], 1) << "Must have at least 1 HDF5 filename listed in "
      << source;


    file_permutation.clear();
    file_permutation.resize(num_files_[n]);
    // Default to identity permutation.
    for (int i = 0; i < num_files_[n]; i++) {
      file_permutation[i] = i;
    }

    // Shuffle if needed.
    if (this->layer_param_.hdf5_data_param2().shuffle()) {
      std::random_shuffle(file_permutation.begin(), file_permutation.end());
    }
    file_permutations_.push_back(file_permutation);

  }



    // Load the first HDF5 file
    LoadHDF5FileData(hdf_filenames_[0][file_permutations_[0][0]].c_str());

    // Reshape blobs.
    // const int batch_size = this->layer_param_.hdf5_data_param2().batch_size();
    const int top_size = this->layer_param_.top_size();
    vector<int> top_shape;
    for (int i = 0; i < top_size; ++i) {
      top_shape.resize(hdf_blobs_[i]->num_axes());
      top_shape[0] = batch_size_;
      for (int j = 1; j < top_shape.size(); ++j) {
        top_shape[j] = hdf_blobs_[i]->shape(j);
      }
      top[i]->Reshape(top_shape);
    }



}

template <typename Dtype>
void HDF5Data2Layer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
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
              * data_dim], &top[j]->mutable_cpu_data()[i * data_dim]);
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(HDF5Data2Layer, Forward);
#endif

INSTANTIATE_CLASS(HDF5Data2Layer);
REGISTER_LAYER_CLASS(HDF5Data2);

}  // namespace caffe
