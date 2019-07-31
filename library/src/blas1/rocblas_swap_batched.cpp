/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "handle.h"
#include "logging.h"
#include "rocblas.h"
#include "utility.h"

namespace
{
    constexpr int NB = 256;

    template <typename T>
    __global__ void swap_kernel_batched(rocblas_int n, T* x[], rocblas_int incx, T* y[], rocblas_int incy)
    {
        ssize_t tid = blockIdx.x * blockDim.x + threadIdx.x; // only dim1

        if(tid < n)
        {
            T* xb = x[blockIdx.y];
            T* yb = y[blockIdx.y];
            if (incx < 0)
              xb -= incx*(n-1);
            if (incy < 0)
              yb -= incy*(n-1);
              
            auto tmp      = yb[tid * incy];
            yb[tid * incy] = xb[tid * incx];
            xb[tid * incx] = tmp;
        }
    }

    template <typename>
    constexpr char rocblas_swap_name[] = "unknown";
    template <>
    constexpr char rocblas_swap_name<float>[] = "rocblas_sswap_batched";
    template <>
    constexpr char rocblas_swap_name<double>[] = "rocblas_dswap_batched";

    template <class T>
    rocblas_status rocblas_swap_batched(
        rocblas_handle handle, rocblas_int n, T* x[], rocblas_int incx, T* y[], rocblas_int incy, rocblas_int batch_count)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        auto layer_mode = handle->layer_mode;
        if(layer_mode & rocblas_layer_mode_log_trace)
            log_trace(handle, rocblas_swap_name<T>, n, x, incx, y, incy);
        if(layer_mode & rocblas_layer_mode_log_bench)
            log_bench(handle,
                      "./rocblas-bench -f swap_batched -r",
                      rocblas_precision_string<T>,
                      "-n",
                      n,
                      "--incx",
                      incx,
                      "--incy",
                      incy,
                      "--batch",
                      batch_count);
        if(layer_mode & rocblas_layer_mode_log_profile)
            log_profile(handle, rocblas_swap_name<T>, "N", n, "incx", incx, "incy", incy);

        if(!x || !y)
            return rocblas_status_invalid_pointer;

        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        // Quick return if possible.
        if(n <= 0)
            return rocblas_status_success;

        hipStream_t rocblas_stream = handle->rocblas_stream;
        int         blocks         = (n - 1) / NB + 1;
        dim3        grid(blocks, batch_count);
        dim3        threads(NB);

        // if(incx < 0)
        //     x -= ptrdiff_t(incx) * (n - 1);
        // if(incy < 0)
        //     y -= ptrdiff_t(incy) * (n - 1);

        hipLaunchKernelGGL(swap_kernel_batched, grid, threads, 0, rocblas_stream, n, x, incx, y, incy);

        return rocblas_status_success;
    }

} // namespace

/* ============================================================================================ */

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_sswap_batched(
    rocblas_handle handle, rocblas_int n, float* x[], rocblas_int incx, float* y[], rocblas_int incy, rocblas_int batch_count)
{
    return rocblas_swap_batched(handle, n, x, incx, y, incy, batch_count);
}

rocblas_status rocblas_dswap_batched(
    rocblas_handle handle, rocblas_int n, double* x[], rocblas_int incx, double* y[], rocblas_int incy, rocblas_int batch_count)
{
    return rocblas_swap_batched(handle, n, x, incx, y, incy, batch_count);
}

} // extern "C"