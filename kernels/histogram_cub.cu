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
constexpr int ITEMS_PER_THREAD = 16;
constexpr int NUM_BINS = 256;


__global__ void histogram_kernel(int* final_histo_output, const unsigned char* d_input_pixels, 
                                 int total_num_pixels, int num_channels, int channel_idx) {
    
    using BlockHist = cub::BlockHistogram<int, BLOCK_SIZE, ITEMS_PER_THREAD, NUM_BINS>;
    __shared__ typename BlockHist::TempStorage temp_storage;
    __shared__ int shared_block_hist[NUM_BINS];

    for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
        shared_block_hist[i] = 0;
    }

    __syncthreads();

    int block_offset = blockIdx.x * BLOCK_SIZE * ITEMS_PER_THREAD;
    int local_pixels[ITEMS_PER_THREAD];

    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int logical_index = block_offset + threadIdx.x + i * BLOCK_SIZE;
        long long memory_index = (long long)logical_index * num_channels + channel_idx;

        if (logical_index < total_num_pixels) {
            local_pixels[i] = d_input_pixels[memory_index];
        } else {
            local_pixels[i] = 0;
        }
    }

    BlockHist(temp_storage).Histogram(local_pixels, shared_block_hist);
    __syncthreads(); 

    if (threadIdx.x < NUM_BINS) {
        int count = shared_block_hist[threadIdx.x];
        if (count > 0) {
            cuda::atomic_ref<int, cuda::thread_scope_device> atomic_ref(final_histo_output[threadIdx.x]);
            atomic_ref.fetch_add(count, cuda::memory_order_relaxed);
        }
    }
}

torch::Tensor histogram_cub(torch::Tensor input_image) {
    nvtxRangePushA("Histogram CUB");

    int n_channels = 1;
    int pixels_per_channel = 0;

    if (input_image.dim() == 2) {
        n_channels = 1;
    } else if (input_image.dim() == 3) {
        n_channels = input_image.size(2); 
    }

    pixels_per_channel = input_image.size(0) * input_image.size(1);

    const unsigned char* ptr_input_data = input_image.data_ptr<unsigned char>();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    auto options = torch::TensorOptions().dtype(torch::kInt32).device(input_image.device()); 
    torch::Tensor output_tensor = torch::zeros({n_channels, NUM_BINS}, options);
    int* ptr_output = output_tensor.data_ptr<int>();

    int threads_per_block = BLOCK_SIZE;
    int items_per_block = BLOCK_SIZE * ITEMS_PER_THREAD;
    int num_blocks = (pixels_per_channel + items_per_block - 1) / items_per_block; 

    for (int channel = 0; channel < n_channels; channel++) {
        int* ptr_channel_output = ptr_output + (channel * NUM_BINS);
    
        histogram_kernel<<<num_blocks, threads_per_block, 0, stream>>>(ptr_channel_output, ptr_input_data, pixels_per_channel, n_channels, channel);
    }

    nvtxRangePop();
    return output_tensor;
}

struct CastToInt {
    __host__ __device__
    int operator()(unsigned char x) const {
        return static_cast<int>(x);
    }
};

