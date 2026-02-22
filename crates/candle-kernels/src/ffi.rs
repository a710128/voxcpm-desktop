use core::ffi::c_void;

#[cfg(candle_disable_moe)]
#[cold]
fn moe_disabled() -> ! {
    // This fork intentionally disables MoE CUDA kernels.
    // Avoid panicking across an extern ABI boundary.
    std::process::abort()
}

#[cfg(candle_disable_moe)]
#[no_mangle]
pub extern "C" fn moe_gemm_wmma(
    _input: *const c_void,
    _weights: *const c_void,
    _sorted_token_ids: *const i32,
    _expert_ids: *const i32,
    _topk_weights: *const f32,
    _output: *mut c_void,
    _expert_counts: *mut i32,
    _expert_offsets: *mut i32,
    _num_experts: i32,
    _topk: i32,
    _size_m: i32,
    _size_n: i32,
    _size_k: i32,
    _dtype: i32,
    _is_prefill: bool,
    _stream: i64,
) {
    moe_disabled()
}

#[cfg(candle_disable_moe)]
#[no_mangle]
pub extern "C" fn moe_gemm_gguf(
    _input: *const f32,
    _weights: *const c_void,
    _sorted_token_ids: *const i32,
    _expert_ids: *const i32,
    _topk_weights: *const f32,
    _output: *mut c_void,
    _num_experts: i32,
    _topk: i32,
    _size_m: i32,
    _size_n: i32,
    _size_k: i32,
    _gguf_dtype: i32,
    _stream: i64,
) {
    moe_disabled()
}

#[cfg(candle_disable_moe)]
#[no_mangle]
pub extern "C" fn moe_gemm_gguf_prefill(
    _input: *const c_void,
    _weights: *const u8,
    _sorted_token_ids: *const i32,
    _expert_ids: *const i32,
    _topk_weights: *const f32,
    _output: *mut c_void,
    _num_experts: i32,
    _topk: i32,
    _size_m: i32,
    _size_n: i32,
    _size_k: i32,
    _input_dtype: i32,
    _gguf_dtype: i32,
    _stream: i64,
) {
    moe_disabled()
}

#[cfg(not(candle_disable_moe))]
#[allow(dead_code)]
extern "C" {
    // for unquntized models
    pub fn moe_gemm_wmma(
        input: *const c_void,         // device pointer [size_m, size_k]
        weights: *const c_void,       // device pointer [num_experts, size_n, size_k]
        sorted_token_ids: *const i32, // device pointer [size_m]
        expert_ids: *const i32,       // host array [size_m] (expert id per sorted token)
        topk_weights: *const f32,
        output: *mut c_void,      // device pointer [size_m, size_n]
        expert_counts: *mut i32,  // pre-allocated buffer [num_experts]
        expert_offsets: *mut i32, // pre-allocated buffer [num_experts + 1]
        num_experts: i32,
        topk: i32,
        size_m: i32,
        size_n: i32,
        size_k: i32,
        dtype: i32, // 0=float16, 1=bf16 (for input/output)
        is_prefill: bool,
        stream: i64,
    );

    pub fn moe_gemm_gguf(
        input: *const f32,      // input [size_m, size_k]
        weights: *const c_void, // weights [num_experts, size_n, size_k]
        sorted_token_ids: *const i32,
        expert_ids: *const i32,
        topk_weights: *const f32, // device ptr or nullptr
        output: *mut c_void,      // float output [size_m, size_n]
        num_experts: i32,
        topk: i32,
        size_m: i32,
        size_n: i32,
        size_k: i32,
        gguf_dtype: i32, // Q8_0: 0, Q4K: 1, Q2K: 2, Q3k: 3,  Q5K: 4, Q6K: 5  (for weights)
        stream: i64,
    );

    pub fn moe_gemm_gguf_prefill(
        input: *const c_void, // input [size_m, size_k]
        weights: *const u8,   // weights [num_experts, size_n, size_k]
        sorted_token_ids: *const i32,
        expert_ids: *const i32,   //must be host ptr
        topk_weights: *const f32, // device ptr or nullptr
        output: *mut c_void,      // float output [size_m, size_n]
        num_experts: i32,
        topk: i32,
        size_m: i32,
        size_n: i32,
        size_k: i32,
        input_dtype: i32, // 0=f16, 1=bf16 (for inputs)
        gguf_dtype: i32,  //Q8_0: 0, Q4K: 1, Q2K: 2, Q3k: 3,  Q5K: 4, Q6K: 5  (for weights)
        stream: i64,
    );
}
