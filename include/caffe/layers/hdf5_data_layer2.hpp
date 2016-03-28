#ifndef CAFFE_HDF5_DATA_LAYER_HPP_
#define CAFFE_HDF5_DATA_LAYER_HPP_

#include "hdf5.h"

#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/base_data_layer.hpp"

namespace caffe {

/**
 * @brief Provides data to the Net from HDF5 files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class HDF5Data2Layer : public Layer<Dtype> {
 public:
  explicit HDF5Data2Layer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual ~HDF5Data2Layer();
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // Data layers should be shared by multiple solvers in parallel
  virtual inline bool ShareInParallel() const { return true; }
  // Data layers have no bottoms, so reshaping is trivial.
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}

  virtual inline const char* type() const { return "HDF5Data2"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  virtual void LoadHDF5FileData(const char* filename);

  std::vector< std::vector<std::string> > hdf_filenames_;
  std::vector<unsigned int> num_files_;
  std::vector<unsigned int> current_files_;
  std::vector<hsize_t> current_rows_;
  std::vector<shared_ptr<Blob<Dtype> > > hdf_blobs_;
  std::vector<unsigned int> data_permutation_;
  std::vector< std::vector<unsigned int> > file_permutations_;
  std::vector<unsigned int> batch_sizes_;
  unsigned int batch_size_;
  unsigned int num_sources_;

};

}  // namespace caffe

#endif  // CAFFE_HDF5_DATA_LAYER_HPP_
