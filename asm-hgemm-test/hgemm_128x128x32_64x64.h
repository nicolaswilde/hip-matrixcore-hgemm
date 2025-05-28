#include "global.h"

__launch_bounds__(256, 4)
__global__ void hgemm_128x128x32_64x64_32x32_naive(
    const half *a, const half *b, float *c, int M, int N, int K) {

    using float16 = float __attribute__((ext_vector_type(16)));
    using half4 = __fp16 __attribute__((ext_vector_type(4)));
    typedef struct {
        half4 x;
        half4 y;
    } half4x2;

    __shared__ half s_a[128][32];
    __shared__ half s_b[128][32];

    int thread_block_m = blockIdx.x * 128;
    int thread_block_n = blockIdx.y * 128;

    int warp_m = (threadIdx.y / 2) * 64;
    int warp_n = (threadIdx.y % 2) * 64;

    int thread_c_m = (threadIdx.x / 32) * 4;
    int thread_c_n = threadIdx.x % 32;

    // LDGSTS A&B Parameters
    int sts_mn = threadIdx.y * 16 + threadIdx.x / 4;
    int sts_k = (threadIdx.x % 4) * 8; // by half
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

    for (int ib = 0; ib < K/32; ib++) {
        half4x2 ldg_a[2], ldg_b[2];
        for (int ii = 0; ii < 2; ii++) {
            ldg_a[ii] = *(half4x2 *)&a[OFFSET(ldg_m + ii*64, ldg_k + ib*32, K)];
            ldg_b[ii] = *(half4x2 *)&b[OFFSET(ldg_n + ii*64, ldg_k + ib*32, K)];
        }

        for (int ii = 0; ii < 2; ii++) {
            *(half4 *)&s_a[sts_mn + ii*64][SWZ_32(sts_mn + ii*64, sts_k    )] = ldg_a[ii].x;
            *(half4 *)&s_a[sts_mn + ii*64][SWZ_32(sts_mn + ii*64, sts_k + 4)] = ldg_a[ii].y;
            *(half4 *)&s_b[sts_mn + ii*64][SWZ_32(sts_mn + ii*64, sts_k    )] = ldg_b[ii].x;
            *(half4 *)&s_b[sts_mn + ii*64][SWZ_32(sts_mn + ii*64, sts_k + 4)] = ldg_b[ii].y;
        }

        __syncthreads();

        half4 tile_a[2][4], tile_b[2][4];
        for (int ii = 0; ii < 2; ii++) {
            for (int kk = 0; kk < 4; kk++) {
                tile_a[ii][kk] = *(half4 *)&s_a[lds_m + ii*32][SWZ_32(lds_m + ii*32, lds_k + kk*8)];
            }
        }
        for (int ii = 0; ii < 2; ii++) {
            for (int kk = 0; kk < 4; kk++) {
                tile_b[ii][kk] = *(half4 *)&s_b[lds_n + ii*32][SWZ_32(lds_n + ii*32, lds_k + kk*8)];
            }
        }
        for (int ii = 0; ii < 2; ii++) {
            for (int jj = 0; jj < 2; jj++) {
                for (int kk = 0; kk < 4; kk++) {
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