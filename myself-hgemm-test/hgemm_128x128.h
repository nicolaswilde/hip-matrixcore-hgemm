#include "global.h"

__launch_bounds__(512, 2)
__global__ void hgemm_128x128_32x64_16x16_swz(
    const half *a, const half *b, float *c, int M, int N, int K) {

    using float4 = float __attribute__((ext_vector_type(4)));
    using half4 = __fp16 __attribute__((ext_vector_type(4)));

    __shared__ half s_a[128][64];
    __shared__ half s_b[128][64];

    int thread_block_m = blockIdx.x * 128;
    int thread_block_n = blockIdx.y * 128;

    int warp_m = (threadIdx.y / 2) * 32;
    int warp_n = (threadIdx.y % 2) * 64;

    int thread_c_m = (threadIdx.x / 16) * 4;
    int thread_c_n = threadIdx.x % 16;

    // LDGSTS A&B Parameters
    int sts_mn = threadIdx.y * 8 + threadIdx.x / 8;
    int sts_k = (threadIdx.x % 8) * 8; // by half
    int ldg_m = thread_block_m + sts_mn;
    int ldg_n = thread_block_n + sts_mn;
    int ldg_k = sts_k;

    // LDS A&B Parameters
    int lds_m = warp_m + threadIdx.x % 16;
    int lds_n = warp_n + threadIdx.x % 16;
    int lds_k = (threadIdx.x / 16) * 4; // by half

    // C&D Parameters
    int c_m = thread_block_m + warp_m + thread_c_m;
    int c_n = thread_block_n + warp_n + thread_c_n;

    float4 tile_d[2][4] = {0};

    for (int ib = 0; ib < K/64; ib++) {
        half ldg_a[2][8], ldg_b[2][8];
        for (int ii = 0; ii < 2; ii++) {
            *(float4 *)ldg_a[ii] = *(float4 *)&a[OFFSET(ldg_m + ii*64, ldg_k + ib*64, K)];
            *(float4 *)ldg_b[ii] = *(float4 *)&b[OFFSET(ldg_n + ii*64, ldg_k + ib*64, K)];
        }

        for (int ii = 0; ii < 2; ii++) {
            *(long *)&s_a[sts_mn + ii*64][SWZ(sts_mn, sts_k    )] = *(long *)&ldg_a[ii][0];
            *(long *)&s_a[sts_mn + ii*64][SWZ(sts_mn, sts_k + 4)] = *(long *)&ldg_a[ii][4];
            *(long *)&s_b[sts_mn + ii*64][SWZ(sts_mn, sts_k    )] = *(long *)&ldg_b[ii][0];
            *(long *)&s_b[sts_mn + ii*64][SWZ(sts_mn, sts_k + 4)] = *(long *)&ldg_b[ii][4];
        }

        __syncthreads();

        half4 tile_a[2][4], tile_b[4][4];
        for (int ii = 0; ii < 2; ii++) {
            for (int kk = 0; kk < 4; kk++) {
                tile_a[ii][kk] = *(long *)&s_a[lds_m + ii*16][SWZ(lds_n, lds_k + kk*16)];
            }
        }
        for (int ii = 0; ii < 4; ii++) {
            for (int kk = 0; kk < 4; kk++) {
                tile_b[ii][kk] = *(long *)&s_b[lds_n + ii*16][SWZ(lds_m, lds_k + kk*16)];
            }
        }
        for (int ii = 0; ii < 2; ii++) {
            for (int jj = 0; jj < 4; jj++) {
                for (int kk = 0; kk < 4; kk++) {
                    tile_d[ii][jj] = __builtin_amdgcn_mfma_f32_16x16x16f16(tile_a[ii][kk], tile_b[jj][kk], tile_d[ii][jj], 0, 0, 0);
                }
            }
        }

        __syncthreads();
    }

    for (int ii = 0; ii < 2; ii++) {
        for (int jj = 0; jj < 4; jj++) {
            for (int kk = 0; kk < 4; kk++) {
                c[(c_m + ii*16 + kk) * N + c_n + jj*16] = tile_d[ii][jj][kk];
            }
        }
    }
}

__launch_bounds__(512, 2)
__global__ void hgemm_128x128_32x64_32x32_swz(
    const half *a, const half *b, float *c, int M, int N, int K) {

    using float16 = float __attribute__((ext_vector_type(16)));
    using float4 = float __attribute__((ext_vector_type(4)));
    using half4 = __fp16 __attribute__((ext_vector_type(4)));

    __shared__ half s_a[128][64];
    __shared__ half s_b[128][64];

    int thread_block_m = blockIdx.x * 128;
    int thread_block_n = blockIdx.y * 128;

    int warp_m = (threadIdx.y / 2) * 32;
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

    float16 tile_d[1][2] = {0};

    for (int ib = 0; ib < K/64; ib++) {
        half ldg_a[2][8], ldg_b[2][8];
        for (int ii = 0; ii < 2; ii++) {
            *(float4 *)ldg_a[ii] = *(float4 *)&a[OFFSET(ldg_m + ii*64, ldg_k + ib*64, K)];
            *(float4 *)ldg_b[ii] = *(float4 *)&b[OFFSET(ldg_n + ii*64, ldg_k + ib*64, K)];
        }

        for (int ii = 0; ii < 2; ii++) {
            *(long *)&s_a[sts_mn + ii*64][SWZ(sts_mn, sts_k    )] = *(long *)&ldg_a[ii][0];
            *(long *)&s_a[sts_mn + ii*64][SWZ(sts_mn, sts_k + 4)] = *(long *)&ldg_a[ii][4];
            *(long *)&s_b[sts_mn + ii*64][SWZ(sts_mn, sts_k    )] = *(long *)&ldg_b[ii][0];
            *(long *)&s_b[sts_mn + ii*64][SWZ(sts_mn, sts_k + 4)] = *(long *)&ldg_b[ii][4];
        }

        __syncthreads();

        half4 tile_a[1][8], tile_b[2][8];
        for (int ii = 0; ii < 1; ii++) {
            for (int kk = 0; kk < 8; kk++) {
                tile_a[ii][kk] = *(long *)&s_a[lds_m + ii*32][SWZ(lds_n, lds_k + kk*8)];
            }
        }
        for (int ii = 0; ii < 2; ii++) {
            for (int kk = 0; kk < 8; kk++) {
                tile_b[ii][kk] = *(long *)&s_b[lds_n + ii*32][SWZ(lds_m, lds_k + kk*8)];
            }
        }
        for (int ii = 0; ii < 1; ii++) {
            for (int jj = 0; jj < 2; jj++) {
                for (int kk = 0; kk < 8; kk++) {
                    tile_d[ii][jj] = __builtin_amdgcn_mfma_f32_32x32x8f16(tile_a[ii][kk], tile_b[jj][kk], tile_d[ii][jj], 0, 0, 0);
                }
            }
        }

        __syncthreads();
    }

    for (int ii = 0; ii < 1; ii++) {
        for (int jj = 0; jj < 2; jj++) {
            for (int kk = 0; kk < 4; kk++) {
                for (int ll = 0; ll < 4; ll++) {
                    c[(c_m + ii*32 + kk*8 + ll) * N + c_n + jj*32] = tile_d[ii][jj][kk*4 + ll];
                }
            }
        }
    }
}