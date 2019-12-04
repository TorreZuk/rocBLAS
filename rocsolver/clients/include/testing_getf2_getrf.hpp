/* ************************************************************************
 * Copyright 2018-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "cblas_interface.hpp"
#include "flops.hpp"
#include "norm.hpp"
#include "rocblas_init.hpp"
#include "rocblas_math.hpp"
#include "rocblas_random.hpp"
#include "rocblas_test.hpp"
#include "rocblas_vector.hpp"
#include "rocsolver.hpp"
#include "unit.hpp"
#include "utility.hpp"

// this is max error PER element after the LU
#define ERROR_EPS_MULTIPLIER 50

using namespace std;

template <typename T, int getrf>
void testing_getf2_getrf(Arguments argus)
{
    rocblas_int    M         = argus.M;
    rocblas_int    N         = argus.N;
    rocblas_int    lda       = argus.lda;
    int            hot_calls = argus.iters;
    rocblas_status status;

    rocblas_local_handle handle;
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);

    // check invalid size and quick return
    if(M < 0 || N < 0 || lda < M)
    {
        device_vector<T>   dA(1);
        device_vector<int> dIpiv(1);
        device_vector<int> dinfo(1);

        if(!dA || !dIpiv || !dinfo)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        if(getrf)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_getrf<T>(handle, M, N, dA, lda, dIpiv, dinfo),
                                  rocblas_status_invalid_size);
        }
        else
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_getf2<T>(handle, M, N, dA, lda, dIpiv, dinfo),
                                  rocblas_status_invalid_size);
        }
        return;
    }

    rocblas_int size_A   = lda * N;
    rocblas_int size_piv = min(M, N);

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T>   hA(size_A);
    host_vector<T>   hAr(size_A);
    host_vector<int> hIpiv(size_piv);
    host_vector<int> hIpivr(size_piv);
    int              hinfo;
    int              hinfor;

    device_vector<T>   dA(size_A);
    device_vector<int> dIpiv(size_piv);
    device_vector<int> dinfo(1);

    if((size_A > 0 && !dA) || (size_piv > 0 && !dIpiv) || !dinfo)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    //initialize full random matrix hA
    rocblas_seedrand();
    rocblas_init<T>(hA, M, N, lda);

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * size_A, hipMemcpyHostToDevice));

    double gpu_time_used, cpu_time_used;
    double error_eps_multiplier = ERROR_EPS_MULTIPLIER;
    double eps                  = std::numeric_limits<T>::epsilon();
    double max_err_1 = 0.0, max_val = 0.0;
    double diff;
    int    piverr = 0;

    /* =====================================================================
           ROCSOLVER
    =================================================================== */
    if(argus.unit_check || argus.norm_check)
    {
        //GPU lapack
        if(getrf)
        {
            CHECK_ROCBLAS_ERROR(rocsolver_getrf<T>(handle, M, N, dA, lda, dIpiv, dinfo));
        }
        else
        {
            CHECK_ROCBLAS_ERROR(rocsolver_getf2<T>(handle, M, N, dA, lda, dIpiv, dinfo));
        }
        //copy output from device to cpu
        CHECK_HIP_ERROR(hipMemcpy(hAr.data(), dA, sizeof(T) * size_A, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(
            hipMemcpy(hIpivr.data(), dIpiv, sizeof(int) * size_piv, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(&hinfor, dinfo, sizeof(int), hipMemcpyDeviceToHost));

        //CPU lapack
        cpu_time_used = get_time_us();
        if(getrf)
        {
            cblas_getrf<T>(M, N, hA.data(), lda, hIpiv.data(), &hinfo);
        }
        else
        {
            cblas_getf2<T>(M, N, hA.data(), lda, hIpiv.data(), &hinfo);
        }
        cpu_time_used = get_time_us() - cpu_time_used;

        // +++++++++ Error Check +++++++++++++
        // check singularity
        if(hinfo != hinfor)
        {
            piverr = 1;
            cerr << "error singular pivot: " << hinfo << " vs " << hinfor << endl;
        }
        // check if the pivoting returned is identical
        for(int j = 0; j < size_piv; j++)
        {
            const int refPiv = hIpiv[j];
            const int gpuPiv = hIpivr[j];
            if(refPiv != gpuPiv)
            {
                piverr = 1;
                cerr << "error reference pivot " << j << ": " << refPiv << " vs " << gpuPiv << endl;
                break;
            }
        }
        // hAr contains calculated decomposition, so error is hA - hAr
        for(int i = 0; i < M; i++)
        {
            for(int j = 0; j < N; j++)
            {
                diff      = fabs(hA[i + j * lda]);
                max_val   = max_val > diff ? max_val : diff;
                diff      = hA[i + j * lda];
                diff      = fabs(hAr[i + j * lda] - diff);
                max_err_1 = max_err_1 > diff ? max_err_1 : diff;
            }
        }
        max_err_1 = max_err_1 / max_val;

        if(argus.unit_check && !piverr)
            err_res_check<T>(max_err_1, max(M, N), error_eps_multiplier, eps);
    }

    if(argus.timing)
    {
        // GPU rocBLAS
        int cold_calls = 2;

        if(getrf)
        {
            for(int iter = 0; iter < cold_calls; iter++)
                rocsolver_getrf<T>(handle, M, N, dA, lda, dIpiv, dinfo);
            gpu_time_used = get_time_us();
            for(int iter = 0; iter < hot_calls; iter++)
                rocsolver_getrf<T>(handle, M, N, dA, lda, dIpiv, dinfo);
            gpu_time_used = (get_time_us() - gpu_time_used) / hot_calls;
        }
        else
        {
            for(int iter = 0; iter < cold_calls; iter++)
                rocsolver_getf2<T>(handle, M, N, dA, lda, dIpiv, dinfo);
            gpu_time_used = get_time_us();
            for(int iter = 0; iter < hot_calls; iter++)
                rocsolver_getf2<T>(handle, M, N, dA, lda, dIpiv, dinfo);
            gpu_time_used = (get_time_us() - gpu_time_used) / hot_calls;
        }

        // only norm_check return an norm error, unit check won't return anything
        cout << "M,N,lda,gpu_time(us),cpu_time(us)";

        if(argus.norm_check)
            cout << ",norm_error_host_ptr";

        cout << endl;
        cout << M << "," << N << "," << lda << "," << gpu_time_used << "," << cpu_time_used;

        if(argus.norm_check)
            cout << "," << max_err_1;

        cout << endl;
    }
}

#undef ERROR_EPS_MULTIPLIER
