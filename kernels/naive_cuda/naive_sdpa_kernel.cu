#include "naive_sdpa_kernel.h"

__global__ void naive_sdpa_kernel(const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V, float* __restrict__ output, float* __restrict__ S_matrix, int batch_size, int seq_len, int num_heads, int head_dim) {
        int batch_idx = blockIdx.x / num_heads;
        int head_idx = blockIdx.x % num_heads;
        int seq_idx = threadIdx.x;
        float max_score = -1e9; // Initialize to a very small value for numerical stability
        float scale = 1.0f / sqrtf((float)head_dim);
        for(int s = 0; s < seq_len; ++s) {
            float dot_product = 0.0f;
            for(int d = 0; d < head_dim; ++d) {
                dot_product += Q[(batch_idx * num_heads * seq_len * head_dim) + (head_idx * seq_len * head_dim) + (seq_idx * head_dim) + d] *
                               K[(batch_idx * num_heads * seq_len * head_dim) + (head_idx * seq_len * head_dim) + (s * head_dim) + d];
            }

            dot_product *= scale;
            
            S_matrix[(batch_idx * num_heads * seq_len * seq_len) + (head_idx * seq_len * seq_len) + (seq_idx * seq_len) + s] = dot_product;
            if(dot_product > max_score) {
                max_score = dot_product;
            }
        }
        float sum_exp = 0.0f;
        for(int s = 0; s < seq_len; ++s){
            S_matrix[(batch_idx * num_heads * seq_len * seq_len) + (head_idx * seq_len * seq_len) + (seq_idx * seq_len) + s] = expf(S_matrix[(batch_idx * num_heads * seq_len * seq_len) + (head_idx * seq_len * seq_len) + (seq_idx * seq_len) + s] - max_score);
            sum_exp += S_matrix[(batch_idx * num_heads * seq_len * seq_len) + (head_idx * seq_len * seq_len) + (seq_idx * seq_len) + s];
        }
        for(int s = 0; s < seq_len; ++s){
            S_matrix[(batch_idx * num_heads * seq_len * seq_len) + (head_idx * seq_len * seq_len) + (seq_idx * seq_len) + s] /= sum_exp;
        }
        for(int d = 0; d < head_dim; ++d) {
            float attn_output = 0.0f;
            for(int s = 0; s < seq_len; ++s) {
                attn_output += S_matrix[(batch_idx * num_heads * seq_len * seq_len) + (head_idx * seq_len * seq_len) + (seq_idx * seq_len) + s] *
                               V[(batch_idx * num_heads * seq_len * head_dim) + (head_idx * seq_len * head_dim) + (s * head_dim) + d];
            }
            output[(batch_idx * num_heads * seq_len * head_dim) + (head_idx * seq_len * head_dim) + (seq_idx * head_dim) + d] = attn_output;
        }
}

at::Tensor naive_sdpa_cuda(at::Tensor Q, at::Tensor K, at::Tensor V) {
    TORCH_CHECK(Q.is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(K.is_cuda(), "K must be a CUDA tensor");
    TORCH_CHECK(V.is_cuda(), "V must be a CUDA tensor");
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous(), "Tensors must be contiguous");

    int batch_size = Q.size(0);
    int num_heads = Q.size(1);
    int seq_len = Q.size(2);
    int head_dim = Q.size(3);

    auto output = at::zeros({batch_size, num_heads, seq_len, head_dim}, Q.options());

    auto S_matrix = at::zeros({batch_size, num_heads, seq_len, seq_len}, Q.options());

    int threads = seq_len;
    int blocks = batch_size * num_heads;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    naive_sdpa_kernel<<<blocks, threads, 0, stream>>>(Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), output.data_ptr<float>(), S_matrix.data_ptr<float>(), batch_size, seq_len, num_heads, head_dim);

    return output;
}