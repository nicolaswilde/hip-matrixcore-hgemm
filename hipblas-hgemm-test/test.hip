#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hip/amd_detail/amd_hip_fp16.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

#define checkHipErrors(func)                                                    \
{                                                                               \
    hipError_t e = (func);                                                      \
    if(e != hipSuccess)                                                         \
        printf ("%s %d HIP: %s\n", __FILE__,  __LINE__, hipGetErrorString(e));  \
}

#define checkHipblasErrors(func)                                                                        \
{                                                                                                       \
    hipblasStatus_t e = (func);                                                                         \
    if (e != HIPBLAS_STATUS_SUCCESS) {                                                                  \
        printf ("%s %d HIP: ", __FILE__,  __LINE__);                                                    \
        switch (e) {                                                                                    \
            case HIPBLAS_STATUS_NOT_INITIALIZED:   printf("HIPBLAS_STATUS_NOT_INITIALIZED\n");   break; \
            case HIPBLAS_STATUS_ALLOC_FAILED:      printf("HIPBLAS_STATUS_ALLOC_FAILED\n");      break; \
            case HIPBLAS_STATUS_INVALID_VALUE:     printf("HIPBLAS_STATUS_INVALID_VALUE\n");     break; \
            case HIPBLAS_STATUS_MAPPING_ERROR:     printf("HIPBLAS_STATUS_MAPPING_ERROR\n");     break; \
            case HIPBLAS_STATUS_EXECUTION_FAILED:  printf("HIPBLAS_STATUS_EXECUTION_FAILED\n");  break; \
            case HIPBLAS_STATUS_INTERNAL_ERROR:    printf("HIPBLAS_STATUS_INTERNAL_ERROR\n");    break; \
            case HIPBLAS_STATUS_NOT_SUPPORTED:     printf("HIPBLAS_STATUS_NOT_SUPPORTED\n");     break; \
            case HIPBLAS_STATUS_ARCH_MISMATCH:     printf("HIPBLAS_STATUS_ARCH_MISMATCH\n");     break; \
            case HIPBLAS_STATUS_HANDLE_IS_NULLPTR: printf("HIPBLAS_STATUS_HANDLE_IS_NULLPTR\n"); break; \
            case HIPBLAS_STATUS_INVALID_ENUM:      printf("HIPBLAS_STATUS_INVALID_ENUM\n");      break; \
            case HIPBLAS_STATUS_UNKNOWN:           printf("HIPBLAS_STATUS_UNKNOWN\n");           break; \
            default: break;                                                                             \
        }                                                                                               \
    }                                                                                                   \
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

void hipblasHgemm(hipblasHandle_t handle,
    const half *a, const half *b, float *c, int M, int N, int K) {

    float alpha = 1.0;
    float beta = 0.0;
    checkHipblasErrors(hipblasGemmEx(
        handle, HIPBLAS_OP_T, HIPBLAS_OP_N, N, M, K,
        &alpha, b, HIPBLAS_R_16F, K, a, HIPBLAS_R_16F, K,
        &beta, c, HIPBLAS_R_32F, N,
        HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
}

float testHipblasHgemmMaxError(int M, int N, int K) {

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
    hipblasHandle_t handle;
    checkHipblasErrors(hipblasCreate(&handle));
    hipblasHgemm(handle, d_a, d_b, d_c, M, N, K);
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
    checkHipblasErrors(hipblasDestroy(handle));

    return max_error;
}

float testHipblasHgemmPerformance(int M, int N, int K) {

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

    hipblasHandle_t handle;
    checkHipblasErrors(hipblasCreate(&handle));

    // warm up
    hipblasHgemm(handle, d_a, d_b, d_c, M, N, K);

    hipEvent_t start, end;
    checkHipErrors(hipEventCreate(&start));
    checkHipErrors(hipEventCreate(&end));
    checkHipErrors(hipEventRecord(start));
    hipblasHgemm(handle, d_a, d_b, d_c, M, N, K);
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
    checkHipblasErrors(hipblasDestroy(handle));

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
        const int M = 128, N = 256, K = 512;
        float max_error = testHipblasHgemmMaxError(M, N, K);
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
            double this_sec = testHipblasHgemmPerformance(M, N, K);
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