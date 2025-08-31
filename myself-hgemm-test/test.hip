#include "global.h"
#include "hgemm_128x128.h"
#include "hgemm_256x256.h"
#include "hgemm_template.h"

#define checkHipErrors(func)                                                    \
{                                                                               \
    hipError_t e = (func);                                                      \
    if(e != hipSuccess)                                                         \
        printf ("%s %d HIP: %s\n", __FILE__,  __LINE__, hipGetErrorString(e));  \
}

void cpuHgemm(
    const half *a, const half *b, float *c, int M, int N, int K) {

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float psum = 0.0;
            for (int k = 0; k < K; k++) {
                psum += (float)a[OFFSET(m, k, K)] * (float)b[OFFSET(n, k, K)];
            }
            c[OFFSET(m, n, N)] = psum;
        }
    }
}

void myselfHgemm(
    const half *a, const half *b, float *c, int M, int N, int K) {

    // hgemm_128x128x64_32x64_16x16_swz<<<dim3(M / 128, N / 128), dim3(64, 8)>>>(a, b, c, M, N, K);
    // hgemm_128x128x64_32x64_32x32_swz<<<dim3(M / 128, N / 128), dim3(64, 8)>>>(a, b, c, M, N, K);
    // hgemm_128x128x64_64x64_32x32_swz<<<dim3(M / 128, N / 128), dim3(64, 4)>>>(a, b, c, M, N, K);
    // hgemm_128x128x32_64x64_32x32_swz<<<dim3(M / 128, N / 128), dim3(64, 4)>>>(a, b, c, M, N, K);
    // hgemm_256x256x64_64x128_32x32_swz<<<dim3(M / 256, N / 256), dim3(64, 8)>>>(a, b, c, M, N, K);
    hgemm_256x256x64_64x128_32x32_swz_db<<<dim3(M / 256, N / 256), dim3(64, 8)>>>(a, b, c, M, N, K);

    // hgemm_swz_wrapper<128, 128, 128, 64, 64, MFMA_32x32x8_F16>(a, b, c, M, N, K);
}

float testMyselfHgemmMaxError(int M, int N, int K) {

    size_t size_a = M * K * sizeof(half);
    size_t size_b = N * K * sizeof(half);
    size_t size_c = M * N * sizeof(float);

    half *h_a, *h_b, *d_a, *d_b;
    float *h_c, *d_c, *h_d_c;
    h_a = (half *)malloc(size_a);
    h_b = (half *)malloc(size_b);
    h_c = (float *)malloc(size_c);
    checkHipErrors(hipMalloc(&d_a, size_a));
    checkHipErrors(hipMalloc(&d_b, size_b));
    checkHipErrors(hipMalloc(&d_c, size_c));
    h_d_c = (float *)malloc(size_c);

    srand(time(0));
    for (int i = 0; i < M * K; i++)
        h_a[i] = (half)(rand() / float(RAND_MAX));
    for (int i = 0; i < N * K; i++)
        h_b[i] = (half)(rand() / float(RAND_MAX));

    cpuHgemm(h_a, h_b, h_c, M, N, K);

    checkHipErrors(hipMemcpy(d_a, h_a, size_a, hipMemcpyHostToDevice));
    checkHipErrors(hipMemcpy(d_b, h_b, size_b, hipMemcpyHostToDevice));
    myselfHgemm(d_a, d_b, d_c, M, N, K);
    checkHipErrors(hipMemcpy(h_d_c, d_c, size_c, hipMemcpyDeviceToHost));

    float max_error = 0.0;
    for (int i = 0; i < M * N; i++) {
        float this_error = abs(h_d_c[i] - h_c[i]);
        if (max_error != max_error || this_error != this_error) // nan
            max_error = -NAN;
        else
            max_error = max(max_error, this_error);
    }

    free(h_a); free(h_b); free(h_c);
    checkHipErrors(hipFree(d_a));
    checkHipErrors(hipFree(d_b));
    checkHipErrors(hipFree(d_c));
    free(h_d_c);

    return max_error;
}

float testMyselfHgemmPerformance(int M, int N, int K) {

    size_t size_a = M * K * sizeof(half);
    size_t size_b = K * N * sizeof(half);
    size_t size_c = M * N * sizeof(float);

    half *d_a, *d_b;
    float *d_c;
    checkHipErrors(hipMalloc(&d_a, size_a));
    checkHipErrors(hipMalloc(&d_b, size_b));
    checkHipErrors(hipMalloc(&d_c, size_c));

    // memset
    checkHipErrors(hipMemset(d_a, 0xf, size_a));
    checkHipErrors(hipMemset(d_b, 0xf, size_b));
    checkHipErrors(hipMemset(d_c, 0xf, size_c));

    // warm up
    myselfHgemm(d_a, d_b, d_c, M, N, K);

    hipEvent_t start, end;
    checkHipErrors(hipEventCreate(&start));
    checkHipErrors(hipEventCreate(&end));
    checkHipErrors(hipEventRecord(start));
    myselfHgemm(d_a, d_b, d_c, M, N, K);
    checkHipErrors(hipEventRecord(end));
    checkHipErrors(hipEventSynchronize(end));

    float msec, sec;
    checkHipErrors(hipEventElapsedTime(&msec, start, end));
    sec = msec / 1000.0;

    checkHipErrors(hipFree(d_a));
    checkHipErrors(hipFree(d_b));
    checkHipErrors(hipFree(d_c));
    checkHipErrors(hipEventDestroy(start));
    checkHipErrors(hipEventDestroy(end));

    return sec;
}

int main () {
    /*
    const int test_num = 8;
    const int M_list[test_num] = {128, 256, 512, 1024, 2048, 4096, 8192, 16384};
    const int N_list[test_num] = {128, 256, 512, 1024, 2048, 4096, 8192, 16384};
    const int K_list[test_num] = {128, 256, 512, 1024, 2048, 4096, 8192, 16384};
    */

    const int test_num = 64;
    int M_list[test_num];
    int N_list[test_num];
    int K_list[test_num];
    for (int i = 0; i < test_num; i++) {
        M_list[i] = 4096; // MI300X 304 CU
        N_list[i] = 4864; // MI300X 304 CU
        K_list[i] = (i + 1) * 256;
    }

    const int test_repeat = 10;

    {
        const int M = 512, N = 1024, K = 256;
        float max_error = testMyselfHgemmMaxError(M, N, K);
        printf("M N K = %6d %6d %6d, ", M, N, K);
        printf("Max Error = %f\n", max_error);
    }

    for (int i = 0; i < test_num; i++) {
        int M = M_list[i], N = N_list[i], K = K_list[i];

        double max_sec = 0.0;
        double min_sec = DBL_MAX;
        double total_sec = 0.0;

        double sec[test_repeat];
        for (int j = 0; j < test_repeat; j++) {
            double this_sec = testMyselfHgemmPerformance(M, N, K);
            sec[j] = this_sec;
            max_sec = max_sec > this_sec ? max_sec : this_sec;
            min_sec = min_sec < this_sec ? min_sec : this_sec;
            total_sec += this_sec;
        }

        double avg_sec = total_sec / test_repeat;
        double avg_Tflops = ((double)M) * N * K * 2 / 1000 / 1000 / 1000 / 1000 / avg_sec;

        printf("M N K = %6d %6d %6d, ", M, N, K);
        printf("Time = %12.8lf %12.8lf %12.8lf s, ", min_sec, avg_sec, max_sec);
        printf("AVG Performance = %8.4lf Tflops, ", avg_Tflops);

        // MI300X: 304 * 4 matrix cores, 512 FP16 FLOPs per cycle, 2.1 GHz
        double MI300X_Tflops = 304 * 4 * 512 * 2.1 / 1000;
        double avg_utilization = avg_Tflops / MI300X_Tflops * 100.0;
        printf("MI300X AVG Util = %5.2lf %%\n", avg_utilization);
    }
}