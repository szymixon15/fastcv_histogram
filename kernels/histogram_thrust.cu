#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/execution_policy.h>
#include <cuda/atomic>
#include <nvtx3/nvToolsExt.h> 


constexpr int BLOCK_SIZE = 256;
constexpr int ITEMS_PER_THREAD = 32;
constexpr int NUM_BINS = 256;

struct CastToInt {
    __host__ __device__
    int operator()(unsigned char x) const {
        return static_cast<int>(x);
    }
};

torch::Tensor histogram_thrust(torch::Tensor input_image) {
    nvtxRangePushA("Histogram Thrust");

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    int n_channels = 1;
    if (input_image.dim() == 2) {
        n_channels = 1;
    } else if (input_image.dim() == 3) {
        n_channels = input_image.size(2); 
    }

    int num_pixels = input_image.size(0) * input_image.size(1);
    
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(input_image.device());
    torch::Tensor output = torch::zeros({n_channels, NUM_BINS}, options);
    int* ptr_output = output.data_ptr<int>();

    for (int k = 0; k < n_channels; k++) {
        
        torch::Tensor data_buffer;

        if (n_channels > 1) {
            data_buffer = input_image.select(2,k).clone();
        } else {
            data_buffer = input_image.clone();
        }
        
        unsigned char* ptr_data = data_buffer.data_ptr<unsigned char>();

        thrust::device_ptr<unsigned char> d_data_ptr(ptr_data);

        thrust::sort(thrust::cuda::par.on(stream), d_data_ptr, d_data_ptr + num_pixels);

        thrust::device_vector<int> unique_keys(NUM_BINS);
        thrust::device_vector<int> counts(NUM_BINS);

        auto iter_begin = thrust::make_transform_iterator(d_data_ptr, CastToInt());
        auto iter_end = thrust::make_transform_iterator(d_data_ptr + num_pixels, CastToInt());

        thrust::pair<thrust::device_vector<int>::iterator, thrust::device_vector<int>::iterator> new_end;
        
        new_end = thrust::reduce_by_key(
            thrust::cuda::par.on(stream),
            iter_begin,
            iter_end,
            thrust::make_constant_iterator(1),
            unique_keys.begin(),
            counts.begin() 
    );

        int num_unique = thrust::distance(unique_keys.begin(), new_end.first);
        
        std::vector<int> h_keys(num_unique);
        std::vector<int> h_counts(num_unique);

        thrust::copy(unique_keys.begin(), unique_keys.begin() + num_unique, h_keys.begin());
        thrust::copy(counts.begin(), counts.begin() + num_unique, h_counts.begin());

        int* ptr_channel_output = ptr_output + (k * NUM_BINS);
        std::vector<int> cpu_hist(NUM_BINS, 0);

        for(int i = 0; i < num_unique; i++) {
            cpu_hist[h_keys[i]] = h_counts[i];
        }

        cudaMemcpyAsync(
            ptr_channel_output,
            cpu_hist.data(),
            NUM_BINS * sizeof(int),
            cudaMemcpyHostToDevice,
            stream
        );
    }

    cudaDeviceSynchronize();
    nvtxRangePop();
    return output;
}