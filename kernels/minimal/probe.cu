#include <decode/decode_attn.cuh>
#include <decode/decode_handler.cuh>

using namespace flashinfer;
using Page = paged_kv_t<PageStorage::kIndices, QKVLayout::kHND, half, int32_t>;

// All buffers are device pointers, FP16, contiguous; batch=1, MHA only.
// The Python caller validates shape/dtype and supplies its current CUDA stream.
extern "C" int quest_score(void* q, void* out, void* metadata, void* indices,
                            void* indptr, int heads, int dim, int page_size,
                            int pages, void* stream) {
    int chunks = (pages + page_size - 1) / page_size;
    Page cache(heads, page_size, dim, 1, 0, (pages - 1) % page_size + 1,
               chunks - 1, (half*)metadata, (int32_t*)indices, (int32_t*)indptr);
    return MaxPossibleSampleWithPagedKVCache<PageStorage::kIndices,
        QKVLayout::kHND, half, half, int32_t>(
        (half*)q, cache, (half*)out, heads, RotaryMode::kNone, (cudaStream_t)stream);
}

extern "C" int quest_plan(void** result, int heads, int dim, int page_size,
                           int budget_without_last, void* stream) {
    auto* handler = new BatchDecodeHandler();
    handler->SetCUDAStream((cudaStream_t)stream);
    int32_t indptr[] = {0, budget_without_last};
    auto status = handler->BeginForward<PageStorage::kIndices, QKVLayout::kHND,
        half, half, int32_t>(indptr, 1, heads, heads, dim, page_size, RotaryMode::kNone);
    if (status != cudaSuccess) { delete handler; return status; }
    *result = handler;
    return cudaSuccess;
}

extern "C" void quest_release(void* handler) {
    auto* plan = (BatchDecodeHandler*)handler;
    plan->EndForward();
    delete plan;
}

extern "C" int quest_decode(void* q, void* out, void* data, void* indices,
                             void* indptr, int heads, int dim, int page_size,
                             int budget_without_last, int last_len,
                             int last_page, void* stream, void* plan) {
    Page cache(heads, page_size, dim, 1, budget_without_last, last_len,
               last_page, (half*)data, (int32_t*)indices, (int32_t*)indptr);
    if (budget_without_last == 0) {
        return BatchDecodeWithPagedKVCache<PageStorage::kIndices, QKVLayout::kHND,
            half, half, int32_t>((half*)q, cache, kv_partition_info_t<int32_t>(),
            (half*)out, nullptr, nullptr, heads, RotaryMode::kNone, 1.f, 1e4f,
            (cudaStream_t)stream);
    }
    return BatchDecodeWithPagedKVCacheWrapper<PageStorage::kIndices, QKVLayout::kHND,
        half, half, int32_t>((BatchDecodeHandler*)plan, (half*)q, cache,
        (half*)out, nullptr, heads, RotaryMode::kNone, 1.f, 1e4f, (cudaStream_t)stream);
}

extern "C" const char* quest_error(int code) {
    return cudaGetErrorString((cudaError_t)code);
}
