#include "global.h"


template<const int BM, const int BN, const int BK, const int WM, const int WN, mfma_t inst, const int TPB>
// thread block tiling: BM, BN, BK
// warp tiling: WM, WN
// instruction: MFMA_32x32x8_F16 or MFMA_16x16x16_F16
// threads per block: TPB
__launch_bounds__(TPB, LDSSIZE / ((BM + BN) * BK * sizeof(half)))
__global__ void hgemm_swz(const half *a, const half *b, float *c, int M, int N, int K) {

    using SWZ = swz<half, BK>;
    using dtype = typename dtype_selector<inst>::type;

    // instruction tiling
    constexpr int IM = im_selector<inst>::value;
    constexpr int IN = in_selector<inst>::value;
    constexpr int IK = ik_selector<inst>::value;

    constexpr int CPT = (inst == MFMA_32x32x8_F16) ? 16 : 4;

    __shared__ half s_a[BM][BK];
    __shared__ half s_b[BN][BK];

    int thread_block_m = blockIdx.x * BM;
    int thread_block_n = blockIdx.y * BN;

    int warp_m = (threadIdx.y / (BN / WN)) * WM;
    int warp_n = (threadIdx.y % (BN / WN)) * WN;

    int thread_c_m = (threadIdx.x / IN) * 4;
    int thread_c_n = threadIdx.x % IN;

    // LDGSTS A&B Parameters
    const int TPR = BK / (16 / sizeof(half)); // LDGSTS thread per row
    const int RPT = TPB / TPR;                // LDGSTS rows per float4 ldg/sts transmit
    int sts_mn = threadIdx.y * TPW / TPR + threadIdx.x / TPR;
    int sts_k = (threadIdx.x % TPR) * (16 / sizeof(half)); // by half
    int ldg_m = thread_block_m + sts_mn;
    int ldg_n = thread_block_n + sts_mn;
    int ldg_k = sts_k;

    // LDS A&B Parameters
    int lds_m = warp_m + threadIdx.x % IM;
    int lds_n = warp_n + threadIdx.x % IM;
    int lds_k = (threadIdx.x / IM) * (8 / sizeof(half)); // by half

    // C&D Parameters
    int c_m = thread_block_m + warp_m + thread_c_m;
    int c_n = thread_block_n + warp_n + thread_c_n;

    dtype tile_d[WM/IM][WN/IM] = {0};

    for (int ib = 0; ib < K/BK; ib++) {
        half ldg_a[BM/RPT][16/sizeof(half)], ldg_b[BN/RPT][16/sizeof(half)];
        for (int ii = 0; ii < BM/RPT; ii++) {
            *(float4 *)ldg_a[ii] = *(float4 *)&a[OFFSET(ldg_m + ii*RPT, ldg_k + ib*BK, K)];
        }
        for (int ii = 0; ii < BN/RPT; ii++) {
            *(float4 *)ldg_b[ii] = *(float4 *)&b[OFFSET(ldg_n + ii*RPT, ldg_k + ib*BK, K)];
        }

        for (int ii = 0; ii < BM/RPT; ii++) {
            *(long *)&s_a[sts_mn + ii*RPT][SWZ(sts_mn, sts_k    )] = *(long *)&ldg_a[ii][0];
            *(long *)&s_a[sts_mn + ii*RPT][SWZ(sts_mn, sts_k + 4)] = *(long *)&ldg_a[ii][4];
        }

        for (int ii = 0; ii < BN/RPT; ii++) {
            *(long *)&s_b[sts_mn + ii*RPT][SWZ(sts_mn, sts_k    )] = *(long *)&ldg_b[ii][0];
            *(long *)&s_b[sts_mn + ii*RPT][SWZ(sts_mn, sts_k + 4)] = *(long *)&ldg_b[ii][4];
        }

        __syncthreads();

        myhalf4 tile_a[WM/IM][BK/IK], tile_b[WN/IN][BK/IK];
        for (int ii = 0; ii < WM/IM; ii++) {
            for (int kk = 0; kk < BK/IK; kk++) {
                tile_a[ii][kk] = *(long *)&s_a[lds_m + ii*IM][SWZ(lds_n, lds_k + kk*(16/sizeof(half)))];
            }
        }
        for (int ii = 0; ii < WN/IN; ii++) {
            for (int kk = 0; kk < BK/IK; kk++) {
                tile_b[ii][kk] = *(long *)&s_b[lds_n + ii*IN][SWZ(lds_m, lds_k + kk*(16/sizeof(half)))];
            }
        }
        for (int ii = 0; ii < WM/IM; ii++) {
            for (int jj = 0; jj < WN/IN; jj++) {
                for (int kk = 0; kk < BK/IK; kk++) {
                    tile_d[ii][jj] = mfma_selector<inst>::call(tile_a[ii][kk], tile_b[jj][kk], tile_d[ii][jj], 0, 0, 0);
                }
            }
        }

        __syncthreads();
    }

    for (int ii = 0; ii < WM/IM; ii++) {
        for (int jj = 0; jj < WN/IN; jj++) {
            for (int kk = 0; kk < CPT/4; kk++) {
                for (int ll = 0; ll < 4; ll++) {
                    c[(c_m + ii*IM + kk*8 + ll) * N + c_n + jj*IN] = tile_d[ii][jj][kk*4 + ll];
                }
            }
        }
    }
}