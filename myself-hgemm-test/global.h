#ifndef GLOBAL_H
#define GLOBAL_H

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_fp16.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define SWZ(m, n) (((m) * 4 + (n)) % 64) // by half
#define SWZ32(m, n) (((m) * 4 + (n)) % 32) // by half

#define TPW 64 // threads per warp
#define LDSSIZE (64*1024) // 64KB in MI300X

using myfloat16 = float __attribute__((ext_vector_type(16)));
using myfloat4  = float __attribute__((ext_vector_type(4)));
using myhalf4   = __fp16 __attribute__((ext_vector_type(4)));

typedef enum {
    MFMA_32x32x8_F16  = 0,
    MFMA_16x16x16_F16 = 1
} mfma_t;

template<mfma_t inst> struct mfma_selector;
template<> struct mfma_selector<MFMA_32x32x8_F16>  {
    static __device__ myfloat16 call(myhalf4 a, myhalf4 b, myfloat16 c, int, int, int) {
        return __builtin_amdgcn_mfma_f32_32x32x8f16(a, b, c, 0, 0, 0);
    }
};
template<> struct mfma_selector<MFMA_16x16x16_F16> {
    static __device__ myfloat4 call(myhalf4 a, myhalf4 b, myfloat4 c, int, int, int) {
        return __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, c, 0, 0, 0);
    }
};

template<typename T, int BK>
__device__ __forceinline__ int swz(int m, int n) {
    static constexpr int swizzle_elements = 8 / sizeof(T);
    static constexpr int shared_memory_row_elements = 128 / sizeof(T);
    static constexpr int swizzle_per_n_matrix_rows =
        BK >= shared_memory_row_elements ? 1 : (shared_memory_row_elements / BK);
    return (m / swizzle_per_n_matrix_rows * swizzle_elements + n) % shared_memory_row_elements;
}

template<mfma_t inst> struct dtype_selector;
template<> struct dtype_selector<MFMA_32x32x8_F16 > { using type = myfloat16; };
template<> struct dtype_selector<MFMA_16x16x16_F16> { using type = myfloat4 ; };

template<mfma_t inst> struct im_selector;
template<> struct im_selector<MFMA_32x32x8_F16 > { static constexpr int value = 32;};
template<> struct im_selector<MFMA_16x16x16_F16> { static constexpr int value = 16;};

template<mfma_t inst> struct in_selector;
template<> struct in_selector<MFMA_32x32x8_F16 > { static constexpr int value = 32;};
template<> struct in_selector<MFMA_16x16x16_F16> { static constexpr int value = 16;};

template<mfma_t inst> struct ik_selector;
template<> struct ik_selector<MFMA_32x32x8_F16 > { static constexpr int value =  8;};
template<> struct ik_selector<MFMA_16x16x16_F16> { static constexpr int value = 16;};

#endif