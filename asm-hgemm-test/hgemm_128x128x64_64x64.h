#include "global.h"

__launch_bounds__(256, 2)
__global__ void hgemm_128x128x64_64x64_32x32_naive(
    const half *a, const half *b, float *c, int M, int N, int K) {

    using float16 = float __attribute__((ext_vector_type(16)));
    using half4 = __fp16 __attribute__((ext_vector_type(4)));
    typedef struct {
        half4 x;
        half4 y;
    } half4x2;

    __shared__ half s_a[128][64];
    __shared__ half s_b[128][64];

    int thread_block_m = blockIdx.x * 128;
    int thread_block_n = blockIdx.y * 128;

    int warp_m = (threadIdx.y / 2) * 64;
    int warp_n = (threadIdx.y % 2) * 64;

    int thread_c_m = (threadIdx.x / 32) * 4;
    int thread_c_n = threadIdx.x % 32;

    // LDGSTS A&B Parameters
    int sts_mn = threadIdx.y * 8 + threadIdx.x / 8;
    int sts_k = (threadIdx.x % 8) * 8; // by half
    int ldg_m = thread_block_m + sts_mn;
    int ldg_n = thread_block_n + sts_mn;
    int ldg_k = sts_k;

    // LDS A&B Parameters
    int lds_m = warp_m + threadIdx.x % 32;
    int lds_n = warp_n + threadIdx.x % 32;
    int lds_k = (threadIdx.x / 32) * 4; // by half

    // C&D Parameters
    int c_m = thread_block_m + warp_m + thread_c_m;
    int c_n = thread_block_n + warp_n + thread_c_n;

    float16 tile_d[2][2] = {0};

    for (int ib = 0; ib < K/64; ib++) {
        half4x2 ldg_a[4], ldg_b[4];
        for (int ii = 0; ii < 4; ii++) {
            ldg_a[ii] = *(half4x2 *)&a[OFFSET(ldg_m + ii*32, ldg_k + ib*64, K)];
            ldg_b[ii] = *(half4x2 *)&b[OFFSET(ldg_n + ii*32, ldg_k + ib*64, K)];
        }

        for (int ii = 0; ii < 4; ii++) {
            *(half4 *)&s_a[sts_mn + ii*32][SWZ(sts_mn, sts_k    )] = ldg_a[ii].x;
            *(half4 *)&s_a[sts_mn + ii*32][SWZ(sts_mn, sts_k + 4)] = ldg_a[ii].y;
            *(half4 *)&s_b[sts_mn + ii*32][SWZ(sts_mn, sts_k    )] = ldg_b[ii].x;
            *(half4 *)&s_b[sts_mn + ii*32][SWZ(sts_mn, sts_k + 4)] = ldg_b[ii].y;
        }

        __syncthreads();

        half4 tile_a[2][8], tile_b[2][8];
        for (int ii = 0; ii < 2; ii++) {
            for (int kk = 0; kk < 8; kk++) {
                tile_a[ii][kk] = *(half4 *)&s_a[lds_m + ii*32][SWZ(lds_m, lds_k + kk*8)];
            }
        }
        for (int ii = 0; ii < 2; ii++) {
            for (int kk = 0; kk < 8; kk++) {
                tile_b[ii][kk] = *(half4 *)&s_b[lds_n + ii*32][SWZ(lds_n, lds_k + kk*8)];
            }
        }
        for (int ii = 0; ii < 2; ii++) {
            for (int jj = 0; jj < 2; jj++) {
                for (int kk = 0; kk < 8; kk++) {
                    tile_d[ii][jj] = __builtin_amdgcn_mfma_f32_32x32x8f16(tile_a[ii][kk], tile_b[jj][kk], tile_d[ii][jj], 0, 0, 0);
                }
            }
        }

        __syncthreads();
    }

    for (int ii = 0; ii < 2; ii++) {
        for (int jj = 0; jj < 2; jj++) {
            for (int kk = 0; kk < 4; kk++) {
                for (int ll = 0; ll < 4; ll++) {
                    c[(c_m + ii*32 + kk*8 + ll) * N + c_n + jj*32] = tile_d[ii][jj][kk*4 + ll];
                }
            }
        }
    }
}

__launch_bounds__(256, 2)
__global__ void hgemm_128x128x64_64x64_32x32_asm_naive(
    const half *a, const half *b, float *c, int M, int N, int K) {

    using float16 = float __attribute__((ext_vector_type(16)));
    using half4 = __fp16 __attribute__((ext_vector_type(4)));
    typedef struct {
        half4 x;
        half4 y;
    } half4x2;

    __shared__ half s_a[128][64];
    __shared__ half s_b[128][64];

    unsigned int s_a_ptr = reinterpret_cast<unsigned long long>(s_a) & 0xFFFFFFFF;
    unsigned int s_b_ptr = reinterpret_cast<unsigned long long>(s_b) & 0xFFFFFFFF;

    int thread_block_m = blockIdx.x * 128;
    int thread_block_n = blockIdx.y * 128;

    int warp_m = (threadIdx.y / 2) * 64;
    int warp_n = (threadIdx.y % 2) * 64;

    int thread_c_m = (threadIdx.x / 32) * 4;
    int thread_c_n = threadIdx.x % 32;

    // LDGSTS A&B Parameters
    int sts_mn = threadIdx.y * 8 + threadIdx.x / 8;
    int sts_k = (threadIdx.x % 8) * 8; // by half
    int ldg_m = thread_block_m + sts_mn;
    int ldg_n = thread_block_n + sts_mn;
    int ldg_k = sts_k;

    // LDS A&B Parameters
    int lds_m = warp_m + threadIdx.x % 32;
    int lds_n = warp_n + threadIdx.x % 32;
    int lds_k = (threadIdx.x / 32) * 4; // by half

    // C&D Parameters
    int c_m = thread_block_m + warp_m + thread_c_m;
    int c_n = thread_block_n + warp_n + thread_c_n;

    float16 tile_d[2][2] = {0};

    for (int ib = 0; ib < K/64; ib++) {
        half4x2 ldg_a[4], ldg_b[4];
        for (int ii = 0; ii < 4; ii++) {
            asm volatile("global_load_dwordx4 %0 %1 off" : "=v"(ldg_a[ii]) : "v"(&a[OFFSET(ldg_m + ii*32, ldg_k + ib*64, K)]));
            asm volatile("global_load_dwordx4 %0 %1 off" : "=v"(ldg_b[ii]) : "v"(&b[OFFSET(ldg_n + ii*32, ldg_k + ib*64, K)]));
        }
        asm volatile("s_waitcnt vmcnt(0)" : : : "memory");
        for (int ii = 0; ii < 4; ii++) {
            asm volatile("ds_write_b64 %0 %1" :: "v"(s_a_ptr + (sts_mn + ii*32) * 128 + SWZ(sts_mn, sts_k    ) * 2), "v"(ldg_a[ii].x));
            asm volatile("ds_write_b64 %0 %1" :: "v"(s_a_ptr + (sts_mn + ii*32) * 128 + SWZ(sts_mn, sts_k + 4) * 2), "v"(ldg_a[ii].y));
            asm volatile("ds_write_b64 %0 %1" :: "v"(s_b_ptr + (sts_mn + ii*32) * 128 + SWZ(sts_mn, sts_k    ) * 2), "v"(ldg_b[ii].x));
            asm volatile("ds_write_b64 %0 %1" :: "v"(s_b_ptr + (sts_mn + ii*32) * 128 + SWZ(sts_mn, sts_k + 4) * 2), "v"(ldg_b[ii].y));
        }

        __syncthreads();

        for (int kk = 0; kk < 8; kk++) {
            half4x2 tile_a, tile_b;
            asm volatile("ds_read2st64_b64 %0, %1 offset1:8" : "=v"(tile_a) : "v"(s_a_ptr + lds_m * 128 + SWZ(lds_m, lds_k + kk*8) * 2));
            asm volatile("ds_read2st64_b64 %0, %1 offset1:8" : "=v"(tile_b) : "v"(s_b_ptr + lds_n * 128 + SWZ(lds_n, lds_k + kk*8) * 2));
            asm volatile("s_waitcnt lgkmcnt(0)" : : : "memory");
            asm volatile("v_mfma_f32_32x32x8f16 %0, %1, %2, %3" : "+v"(tile_d[0][0]) : "v"(tile_a.x), "v"(tile_b.x), "v"(tile_d[0][0]));
            asm volatile("v_mfma_f32_32x32x8f16 %0, %1, %2, %3" : "+v"(tile_d[0][1]) : "v"(tile_a.x), "v"(tile_b.y), "v"(tile_d[0][1]));
            asm volatile("v_mfma_f32_32x32x8f16 %0, %1, %2, %3" : "+v"(tile_d[1][0]) : "v"(tile_a.y), "v"(tile_b.x), "v"(tile_d[1][0]));
            asm volatile("v_mfma_f32_32x32x8f16 %0, %1, %2, %3" : "+v"(tile_d[1][1]) : "v"(tile_a.y), "v"(tile_b.y), "v"(tile_d[1][1]));
        }
/*
        for (int kk = 0; kk < 8; kk++) {
            half4x2 tile_a, tile_b;
            asm volatile("ds_read_b64 %0, %1 \n" : "=v"(tile_a.x) : "v"(s_a_ptr + (lds_m     ) * 128 + SWZ(lds_m, lds_k + kk*8) * 2));
            asm volatile("ds_read_b64 %0, %1 \n" : "=v"(tile_a.y) : "v"(s_a_ptr + (lds_m + 32) * 128 + SWZ(lds_m, lds_k + kk*8) * 2));
            asm volatile("ds_read_b64 %0, %1 \n" : "=v"(tile_b.x) : "v"(s_b_ptr + (lds_n     ) * 128 + SWZ(lds_n, lds_k + kk*8) * 2));
            asm volatile("ds_read_b64 %0, %1 \n" : "=v"(tile_b.y) : "v"(s_b_ptr + (lds_n + 32) * 128 + SWZ(lds_n, lds_k + kk*8) * 2));
            asm volatile("s_waitcnt lgkmcnt(0)\n" : : : "memory");
            asm volatile("v_mfma_f32_32x32x8f16 %0, %1, %2, %3" : "+v"(tile_d[0][0]) : "v"(tile_a.x), "v"(tile_b.x), "v"(tile_d[0][0]));
            asm volatile("v_mfma_f32_32x32x8f16 %0, %1, %2, %3" : "+v"(tile_d[0][1]) : "v"(tile_a.x), "v"(tile_b.y), "v"(tile_d[0][1]));
            asm volatile("v_mfma_f32_32x32x8f16 %0, %1, %2, %3" : "+v"(tile_d[1][0]) : "v"(tile_a.y), "v"(tile_b.x), "v"(tile_d[1][0]));
            asm volatile("v_mfma_f32_32x32x8f16 %0, %1, %2, %3" : "+v"(tile_d[1][1]) : "v"(tile_a.y), "v"(tile_b.y), "v"(tile_d[1][1]));
        }
*/
        __syncthreads();
    }

    for (int ii = 0; ii < 2; ii++) {
        for (int jj = 0; jj < 2; jj++) {
            for (int kk = 0; kk < 4; kk++) {
                for (int ll = 0; ll < 4; ll++) {
                    c[(c_m + ii*32 + kk*8 + ll) * N + c_n + jj*32] = tile_d[ii][jj][kk*4 + ll];
                }
            }
        }
    }
}

__launch_bounds__(256, 2)
__global__ void hgemm_128x128x64_64x64_32x32_asm(
    const half *a, const half *b, float *c, int M, int N, int K) {

    using float16 = float __attribute__((ext_vector_type(16)));
    using float4 = float __attribute__((ext_vector_type(4)));
    using half4 = __fp16 __attribute__((ext_vector_type(4)));
    typedef struct {
        half4 x;
        half4 y;
    } half4x2;

    __shared__ half s_a[128][64];
    __shared__ half s_b[128][64];

    unsigned int s_a_ptr = reinterpret_cast<unsigned long long>(s_a) & 0xFFFFFFFF;
    unsigned int s_b_ptr = reinterpret_cast<unsigned long long>(s_b) & 0xFFFFFFFF;

    int thread_block_m = blockIdx.x * 128;
    int thread_block_n = blockIdx.y * 128;

    int warp_m = (threadIdx.y / 2) * 64;
    int warp_n = (threadIdx.y % 2) * 64;

    int thread_c_m = (threadIdx.x / 32) * 4;
    int thread_c_n = threadIdx.x % 32;

    // LDGSTS A&B Parameters
    int sts_mn = threadIdx.y * 8 + threadIdx.x / 8;
    int sts_k = (threadIdx.x % 8) * 8; // by half
    int ldg_m = thread_block_m + sts_mn;
    int ldg_n = thread_block_n + sts_mn;
    int ldg_k = sts_k;

    // LDS A&B Parameters
    int lds_m = warp_m + threadIdx.x % 32;
    int lds_n = warp_n + threadIdx.x % 32;
    int lds_k = (threadIdx.x / 32) * 4; // by half

    // C&D Parameters
    int c_m = thread_block_m + warp_m + thread_c_m;
    int c_n = thread_block_n + warp_n + thread_c_n;

    float16 tile_d[2][2] = {0};

    half4x2 *ldg_a_ptr[4], *ldg_b_ptr[4];
    for (int ii = 0; ii < 4; ii++) {
        ldg_a_ptr[ii] = (half4x2 *)&a[OFFSET(ldg_m + ii*32, ldg_k, K)];
        ldg_b_ptr[ii] = (half4x2 *)&b[OFFSET(ldg_n + ii*32, ldg_k, K)];
    }

    unsigned int sts_a_ptr[2], sts_b_ptr[2];
    for (int ii = 0; ii < 2; ii++) {
        sts_a_ptr[ii] = s_a_ptr + sts_mn * 128 + SWZ(sts_mn, sts_k + ii*4) * 2;
        sts_b_ptr[ii] = s_b_ptr + sts_mn * 128 + SWZ(sts_mn, sts_k + ii*4) * 2;
    }

    unsigned int lds_a_ptr[8], lds_b_ptr[8];
    for (int ii = 0; ii < 8; ii++) {
        lds_a_ptr[ii] = s_a_ptr + lds_m * 128 + SWZ(lds_m, lds_k + ii*8) * 2;
        lds_b_ptr[ii] = s_b_ptr + lds_n * 128 + SWZ(lds_n, lds_k + ii*8) * 2;
    }

    for (int ib = 0; ib < K/64; ib++) {

        half4x2 ldg_a[4], ldg_b[4];
        asm volatile("global_load_dwordx4 %0 %1 off" : "=v"(ldg_a[0]) : "v"(ldg_a_ptr[0]));
        asm volatile("global_load_dwordx4 %0 %1 off" : "=v"(ldg_a[1]) : "v"(ldg_a_ptr[1]));
        asm volatile("global_load_dwordx4 %0 %1 off" : "=v"(ldg_a[2]) : "v"(ldg_a_ptr[2]));
        asm volatile("global_load_dwordx4 %0 %1 off" : "=v"(ldg_a[3]) : "v"(ldg_a_ptr[3]));
        asm volatile("global_load_dwordx4 %0 %1 off" : "=v"(ldg_b[0]) : "v"(ldg_b_ptr[0]));
        asm volatile("global_load_dwordx4 %0 %1 off" : "=v"(ldg_b[1]) : "v"(ldg_b_ptr[1]));
        asm volatile("global_load_dwordx4 %0 %1 off" : "=v"(ldg_b[2]) : "v"(ldg_b_ptr[2]));
        asm volatile("global_load_dwordx4 %0 %1 off" : "=v"(ldg_b[3]) : "v"(ldg_b_ptr[3]));
        for (int ii = 0; ii < 4; ii++) {
            ldg_a_ptr[ii] += 8;
            ldg_b_ptr[ii] += 8;
        }

        asm volatile("s_waitcnt vmcnt(7)" : : : "memory");
        asm volatile("ds_write_b64 %0 %1" :: "v"(sts_a_ptr[0]), "v"(ldg_a[0].x));
        asm volatile("ds_write_b64 %0 %1" :: "v"(sts_a_ptr[1]), "v"(ldg_a[0].y));
        asm volatile("s_waitcnt vmcnt(6)" : : : "memory");
        asm volatile("ds_write_b64 %0 %1 offset:4096" :: "v"(sts_a_ptr[0]), "v"(ldg_a[1].x));
        asm volatile("ds_write_b64 %0 %1 offset:4096" :: "v"(sts_a_ptr[1]), "v"(ldg_a[1].y));
        asm volatile("s_waitcnt vmcnt(5)" : : : "memory");
        asm volatile("ds_write_b64 %0 %1 offset:8192" :: "v"(sts_a_ptr[0]), "v"(ldg_a[2].x));
        asm volatile("ds_write_b64 %0 %1 offset:8192" :: "v"(sts_a_ptr[1]), "v"(ldg_a[2].y));
        asm volatile("s_waitcnt vmcnt(4)" : : : "memory");
        asm volatile("ds_write_b64 %0 %1 offset:12288" :: "v"(sts_a_ptr[0]), "v"(ldg_a[3].x));
        asm volatile("ds_write_b64 %0 %1 offset:12288" :: "v"(sts_a_ptr[1]), "v"(ldg_a[3].y));

        asm volatile("s_waitcnt vmcnt(3)" : : : "memory");
        asm volatile("ds_write_b64 %0 %1" :: "v"(sts_b_ptr[0]), "v"(ldg_b[0].x));
        asm volatile("ds_write_b64 %0 %1" :: "v"(sts_b_ptr[1]), "v"(ldg_b[0].y));
        asm volatile("s_waitcnt vmcnt(2)" : : : "memory");
        asm volatile("ds_write_b64 %0 %1 offset:4096" :: "v"(sts_b_ptr[0]), "v"(ldg_b[1].x));
        asm volatile("ds_write_b64 %0 %1 offset:4096" :: "v"(sts_b_ptr[1]), "v"(ldg_b[1].y));
        asm volatile("s_waitcnt vmcnt(1)" : : : "memory");
        asm volatile("ds_write_b64 %0 %1 offset:8192" :: "v"(sts_b_ptr[0]), "v"(ldg_b[2].x));
        asm volatile("ds_write_b64 %0 %1 offset:8192" :: "v"(sts_b_ptr[1]), "v"(ldg_b[2].y));
        asm volatile("s_waitcnt vmcnt(0)" : : : "memory");
        asm volatile("ds_write_b64 %0 %1 offset:12288" :: "v"(sts_b_ptr[0]), "v"(ldg_b[3].x));
        asm volatile("ds_write_b64 %0 %1 offset:12288" :: "v"(sts_b_ptr[1]), "v"(ldg_b[3].y));

        __syncthreads();

        half4x2 tile_a0, tile_a1, tile_b0, tile_b1;

        asm volatile("ds_read2st64_b64 %0, %1 offset1:8\n" : "=v"(tile_a0) : "v"(lds_a_ptr[0]));
        asm volatile("ds_read2st64_b64 %0, %1 offset1:8\n" : "=v"(tile_b0) : "v"(lds_b_ptr[0]));

        asm volatile("s_waitcnt lgkmcnt(0)\n" : : : "memory");
        asm volatile("v_mfma_f32_32x32x8f16 %0, %1, %2, %3" : "+v"(tile_d[0][0]) : "v"(tile_a0.x), "v"(tile_b0.x), "v"(tile_d[0][0]));
        asm volatile("ds_read2st64_b64 %0, %1 offset1:8\n" : "=v"(tile_a1) : "v"(lds_a_ptr[1]));
        asm volatile("ds_read2st64_b64 %0, %1 offset1:8\n" : "=v"(tile_b1) : "v"(lds_b_ptr[1]));
        asm volatile("v_mfma_f32_32x32x8f16 %0, %1, %2, %3" : "+v"(tile_d[0][1]) : "v"(tile_a0.x), "v"(tile_b0.y), "v"(tile_d[0][1]));
        asm volatile("v_mfma_f32_32x32x8f16 %0, %1, %2, %3" : "+v"(tile_d[1][0]) : "v"(tile_a0.y), "v"(tile_b0.x), "v"(tile_d[1][0]));
        asm volatile("v_mfma_f32_32x32x8f16 %0, %1, %2, %3" : "+v"(tile_d[1][1]) : "v"(tile_a0.y), "v"(tile_b0.y), "v"(tile_d[1][1]));

        asm volatile("s_waitcnt lgkmcnt(0)\n" : : : "memory");
        asm volatile("v_mfma_f32_32x32x8f16 %0, %1, %2, %3" : "+v"(tile_d[0][0]) : "v"(tile_a1.x), "v"(tile_b1.x), "v"(tile_d[0][0]));
        asm volatile("ds_read2st64_b64 %0, %1 offset1:8\n" : "=v"(tile_a0) : "v"(lds_a_ptr[2]));
        asm volatile("ds_read2st64_b64 %0, %1 offset1:8\n" : "=v"(tile_b0) : "v"(lds_b_ptr[2]));
        asm volatile("v_mfma_f32_32x32x8f16 %0, %1, %2, %3" : "+v"(tile_d[0][1]) : "v"(tile_a1.x), "v"(tile_b1.y), "v"(tile_d[0][1]));
        asm volatile("v_mfma_f32_32x32x8f16 %0, %1, %2, %3" : "+v"(tile_d[1][0]) : "v"(tile_a1.y), "v"(tile_b1.x), "v"(tile_d[1][0]));
        asm volatile("v_mfma_f32_32x32x8f16 %0, %1, %2, %3" : "+v"(tile_d[1][1]) : "v"(tile_a1.y), "v"(tile_b1.y), "v"(tile_d[1][1]));

        asm volatile("s_waitcnt lgkmcnt(0)\n" : : : "memory");
        asm volatile("v_mfma_f32_32x32x8f16 %0, %1, %2, %3" : "+v"(tile_d[0][0]) : "v"(tile_a0.x), "v"(tile_b0.x), "v"(tile_d[0][0]));
        asm volatile("ds_read2st64_b64 %0, %1 offset1:8\n" : "=v"(tile_a1) : "v"(lds_a_ptr[3]));
        asm volatile("ds_read2st64_b64 %0, %1 offset1:8\n" : "=v"(tile_b1) : "v"(lds_b_ptr[3]));
        asm volatile("v_mfma_f32_32x32x8f16 %0, %1, %2, %3" : "+v"(tile_d[0][1]) : "v"(tile_a0.x), "v"(tile_b0.y), "v"(tile_d[0][1]));
        asm volatile("v_mfma_f32_32x32x8f16 %0, %1, %2, %3" : "+v"(tile_d[1][0]) : "v"(tile_a0.y), "v"(tile_b0.x), "v"(tile_d[1][0]));
        asm volatile("v_mfma_f32_32x32x8f16 %0, %1, %2, %3" : "+v"(tile_d[1][1]) : "v"(tile_a0.y), "v"(tile_b0.y), "v"(tile_d[1][1]));

        asm volatile("s_waitcnt lgkmcnt(0)\n" : : : "memory");
        asm volatile("v_mfma_f32_32x32x8f16 %0, %1, %2, %3" : "+v"(tile_d[0][0]) : "v"(tile_a1.x), "v"(tile_b1.x), "v"(tile_d[0][0]));
        asm volatile("ds_read2st64_b64 %0, %1 offset1:8\n" : "=v"(tile_a0) : "v"(lds_a_ptr[4]));
        asm volatile("ds_read2st64_b64 %0, %1 offset1:8\n" : "=v"(tile_b0) : "v"(lds_b_ptr[4]));
        asm volatile("v_mfma_f32_32x32x8f16 %0, %1, %2, %3" : "+v"(tile_d[0][1]) : "v"(tile_a1.x), "v"(tile_b1.y), "v"(tile_d[0][1]));
        asm volatile("v_mfma_f32_32x32x8f16 %0, %1, %2, %3" : "+v"(tile_d[1][0]) : "v"(tile_a1.y), "v"(tile_b1.x), "v"(tile_d[1][0]));
        asm volatile("v_mfma_f32_32x32x8f16 %0, %1, %2, %3" : "+v"(tile_d[1][1]) : "v"(tile_a1.y), "v"(tile_b1.y), "v"(tile_d[1][1]));

        asm volatile("s_waitcnt lgkmcnt(0)\n" : : : "memory");
        asm volatile("v_mfma_f32_32x32x8f16 %0, %1, %2, %3" : "+v"(tile_d[0][0]) : "v"(tile_a0.x), "v"(tile_b0.x), "v"(tile_d[0][0]));
        asm volatile("ds_read2st64_b64 %0, %1 offset1:8\n" : "=v"(tile_a1) : "v"(lds_a_ptr[5]));
        asm volatile("ds_read2st64_b64 %0, %1 offset1:8\n" : "=v"(tile_b1) : "v"(lds_b_ptr[5]));
        asm volatile("v_mfma_f32_32x32x8f16 %0, %1, %2, %3" : "+v"(tile_d[0][1]) : "v"(tile_a0.x), "v"(tile_b0.y), "v"(tile_d[0][1]));
        asm volatile("v_mfma_f32_32x32x8f16 %0, %1, %2, %3" : "+v"(tile_d[1][0]) : "v"(tile_a0.y), "v"(tile_b0.x), "v"(tile_d[1][0]));
        asm volatile("v_mfma_f32_32x32x8f16 %0, %1, %2, %3" : "+v"(tile_d[1][1]) : "v"(tile_a0.y), "v"(tile_b0.y), "v"(tile_d[1][1]));

        asm volatile("s_waitcnt lgkmcnt(0)\n" : : : "memory");
        asm volatile("v_mfma_f32_32x32x8f16 %0, %1, %2, %3" : "+v"(tile_d[0][0]) : "v"(tile_a1.x), "v"(tile_b1.x), "v"(tile_d[0][0]));
        asm volatile("ds_read2st64_b64 %0, %1 offset1:8\n" : "=v"(tile_a0) : "v"(lds_a_ptr[6]));
        asm volatile("ds_read2st64_b64 %0, %1 offset1:8\n" : "=v"(tile_b0) : "v"(lds_b_ptr[6]));
        asm volatile("v_mfma_f32_32x32x8f16 %0, %1, %2, %3" : "+v"(tile_d[0][1]) : "v"(tile_a1.x), "v"(tile_b1.y), "v"(tile_d[0][1]));
        asm volatile("v_mfma_f32_32x32x8f16 %0, %1, %2, %3" : "+v"(tile_d[1][0]) : "v"(tile_a1.y), "v"(tile_b1.x), "v"(tile_d[1][0]));
        asm volatile("v_mfma_f32_32x32x8f16 %0, %1, %2, %3" : "+v"(tile_d[1][1]) : "v"(tile_a1.y), "v"(tile_b1.y), "v"(tile_d[1][1]));

        asm volatile("s_waitcnt lgkmcnt(0)\n" : : : "memory");
        asm volatile("v_mfma_f32_32x32x8f16 %0, %1, %2, %3" : "+v"(tile_d[0][0]) : "v"(tile_a0.x), "v"(tile_b0.x), "v"(tile_d[0][0]));
        asm volatile("ds_read2st64_b64 %0, %1 offset1:8\n" : "=v"(tile_a1) : "v"(lds_a_ptr[7]));
        asm volatile("ds_read2st64_b64 %0, %1 offset1:8\n" : "=v"(tile_b1) : "v"(lds_b_ptr[7]));
        asm volatile("v_mfma_f32_32x32x8f16 %0, %1, %2, %3" : "+v"(tile_d[0][1]) : "v"(tile_a0.x), "v"(tile_b0.y), "v"(tile_d[0][1]));
        asm volatile("v_mfma_f32_32x32x8f16 %0, %1, %2, %3" : "+v"(tile_d[1][0]) : "v"(tile_a0.y), "v"(tile_b0.x), "v"(tile_d[1][0]));
        asm volatile("v_mfma_f32_32x32x8f16 %0, %1, %2, %3" : "+v"(tile_d[1][1]) : "v"(tile_a0.y), "v"(tile_b0.y), "v"(tile_d[1][1]));

        asm volatile("s_waitcnt lgkmcnt(0)\n" : : : "memory");
        asm volatile("v_mfma_f32_32x32x8f16 %0, %1, %2, %3" : "+v"(tile_d[0][0]) : "v"(tile_a1.x), "v"(tile_b1.x), "v"(tile_d[0][0]));
        asm volatile("v_mfma_f32_32x32x8f16 %0, %1, %2, %3" : "+v"(tile_d[0][1]) : "v"(tile_a1.x), "v"(tile_b1.y), "v"(tile_d[0][1]));
        asm volatile("v_mfma_f32_32x32x8f16 %0, %1, %2, %3" : "+v"(tile_d[1][0]) : "v"(tile_a1.y), "v"(tile_b1.x), "v"(tile_d[1][0]));
        asm volatile("v_mfma_f32_32x32x8f16 %0, %1, %2, %3" : "+v"(tile_d[1][1]) : "v"(tile_a1.y), "v"(tile_b1.y), "v"(tile_d[1][1]));

        __syncthreads();
    }

    for (int ii = 0; ii < 2; ii++) {
        for (int jj = 0; jj < 2; jj++) {
            for (int kk = 0; kk < 4; kk++) {
                for (int ll = 0; ll < 4; ll++) {
                    c[(c_m + ii*32 + kk*8 + ll) * N + c_n + jj*32] = tile_d[ii][jj][kk*4 + ll];
                }
            }
        }
    }
}