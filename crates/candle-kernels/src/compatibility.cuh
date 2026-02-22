#include "cuda_fp16.h"
#include "cuda_bf16.h"
#include "cuda_fp8.h"

// Table showing which features are supported on which compute capability
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/#features-and-technical-specifications

// FIXME: the minimum compute capabilities are just guesses since the table is not specific enough

#if (__CUDACC_VER_MAJOR__ < 12 || __CUDACC_VER_MINOR__ < 2) && __CUDA_ARCH__ < 750
__device__ __forceinline__ __half __hmax_nan(__half a, __half b) {
    return __hisnan(a) ? a : (__hisnan(b) ? b : __hmax(a, b));
}
__device__ __forceinline__ __half __hmin_nan(__half a, __half b) {
    return __hisnan(a) ? a : (__hisnan(b) ? b : __hmin(a, b));
}
#endif

#if __CUDA_ARCH__ < 600
// Copied from https://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomic-functions
__device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

#if __CUDA_ARCH__ < 700
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomicadd
// The 16-bit __half floating-point version of atomicAdd() is only supported by devices of compute capability 7.x and higher.
// Solution adapted from https://github.com/torch/cutorch/blob/master/lib/THC/THCAtomics.cuh#L96-L119
__device__ __forceinline__ __half atomicAdd(__half *address, __half val) {
    // Pre-sm_70 there is no native atomicAdd(__half*). Implement via atomicCAS on the
    // containing 32-bit word, updating one 16-bit lane.
    unsigned int *address_as_ui = (unsigned int *)((char *)address - ((size_t)address & 2));
    unsigned int old = *address_as_ui;
    unsigned int assumed;
    bool unaligned = (size_t)address & 2;
    do {
        assumed = old;
        unsigned short old_bits = (unsigned short)(unaligned ? (assumed >> 16) : (assumed & 0xffff));
        __half old_h = __ushort_as_half(old_bits);

        // Do the math in f32 to avoid relying on half ALU details.
        float sum = __half2float(old_h) + __half2float(val);
        __half new_h = __float2half_rn(sum);
        unsigned short new_bits = __half_as_ushort(new_h);

        unsigned int new_word = unaligned ? ((assumed & 0xffff) | ((unsigned int)new_bits << 16))
                                          : ((assumed & 0xffff0000) | (unsigned int)new_bits);
        old = atomicCAS(address_as_ui, assumed, new_word);

        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    unsigned short ret_bits = (unsigned short)(unaligned ? (old >> 16) : (old & 0xffff));
    return __ushort_as_half(ret_bits);
}
#endif


__device__ __forceinline__ __half atomicMaxf(__half* address, __half val) {
#if __CUDA_ARCH__ < 700
    // On older GPUs we do not have access to atomicCAS for shorts, so we have to do some trickery.
    // Solution adapted from https://github.com/torch/cutorch/blob/master/lib/THC/THCAtomics.cuh#L96-L119
    unsigned int *address_as_ui = (unsigned int *) ((char *)address - ((size_t)address & 2));
    unsigned int old = *address_as_ui;
    unsigned int assumed;
    bool unaligned = (size_t) address & 2;
    do {
        assumed = old;
        unsigned int hmax;
        hmax = unaligned ? (old >> 16) : (old & 0xffff);
        hmax = __half_as_ushort(__hmax_nan(val, __ushort_as_half(hmax))); 
        old = atomicCAS(address_as_ui, assumed,
            unaligned ? (old & 0xffff) | (hmax << 16) : (old & 0xffff0000) | hmax
        );

    } while (assumed != old);
    return __ushort_as_half(unaligned ? (old >> 16) : (old & 0xffff));
#else
    // Based on https://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomic-functions
    unsigned short int* casted_address = (unsigned short int*)address;
    unsigned short int old = *casted_address;
    unsigned short int assumed;
    do {
        assumed = old;
        old = atomicCAS(casted_address, assumed, __half_as_ushort(__hmax_nan(val, __ushort_as_half(assumed))));
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);
    return __ushort_as_half(old);
#endif
}

// atomicMax is not implemented for floats,
// solution copied https://stackoverflow.com/questions/17399119/how-do-i-use-atomicmax-on-floating-point-values-in-cuda
__device__ __forceinline__ float atomicMaxf(float * addr, float value) {
    if (signbit(value)) {
        return __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));        
    } else {
        return __int_as_float(atomicMax((int *)addr, __float_as_int(value)));
    }
}

__device__ __forceinline__ double atomicMaxf(double * addr, double value) {
    if (signbit(value)) {
        return __longlong_as_double(atomicMin((unsigned long long int *)addr, __double_as_longlong(value)));
    } else {
        return __longlong_as_double(atomicMax((long long int *)addr, __double_as_longlong(value)));
    }
}


__device__ __forceinline__ __half atomicMinf(__half* address, __half val) {
#if __CUDA_ARCH__ < 700
    // On older GPUs we do not have access to atomicCAS for shorts, so we have to do some trickery.
    // Solution adapted from https://github.com/torch/cutorch/blob/master/lib/THC/THCAtomics.cuh#L96-L119
    unsigned int *address_as_ui = (unsigned int *) ((char *)address - ((size_t)address & 2));
    unsigned int old = *address_as_ui;
    unsigned int assumed;
    bool unaligned = (size_t) address & 2;
    do {
        assumed = old;
        unsigned int hmin;
        hmin = unaligned ? (old >> 16) : (old & 0xffff);
        hmin = __half_as_ushort(__hmin_nan(val, __ushort_as_half(hmin))); 
        old = atomicCAS(address_as_ui, assumed,
            unaligned ? (old & 0xffff) | (hmin << 16) : (old & 0xffff0000) | hmin
        );

    } while (assumed != old);
    return __ushort_as_half(unaligned ? (old >> 16) : (old & 0xffff));
#else
    // Based on https://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomic-functions
    unsigned short int* casted_address = (unsigned short int*)address;
    unsigned short int old = *casted_address;
    unsigned short int assumed;
    do {
        assumed = old;
        old = atomicCAS(casted_address, assumed, __half_as_ushort(__hmin_nan(val, __ushort_as_half(assumed))));
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);
    return __ushort_as_half(old);
#endif
}

// atomicMin is not implemented for floats,
// solution copied https://stackoverflow.com/questions/17399119/how-do-i-use-atomicmax-on-floating-point-values-in-cuda
__device__ __forceinline__ float atomicMinf(float * addr, float value) {
    if (signbit(value)) {
        return __uint_as_float(atomicMax((unsigned int *)addr, __float_as_uint(value)));
    } else {
        return __int_as_float(atomicMin((int *)addr, __float_as_int(value)));
    }
}

__device__ __forceinline__ double atomicMinf(double * addr, double value) {
    if (signbit(value)) {
        return __longlong_as_double(atomicMax((unsigned long long int *)addr, __double_as_longlong(value)));
    } else {
        return __longlong_as_double(atomicMin((long long int *)addr, __double_as_longlong(value)));
    }
}
