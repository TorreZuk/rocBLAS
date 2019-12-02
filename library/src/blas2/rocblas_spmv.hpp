/* ************************************************************************
 * Copyright 2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#ifndef __ROCBLAS_SPMV_HPP__
#define __ROCBLAS_SPMV_HPP__
#include "handle.h"
#include "rocblas.h"

template <typename T, typename U, typename W>
__global__ void rocblas_spmv_kernel(rocblas_fill   uplo,
                                    rocblas_int    n,
                                    V              alpha_device_host,
                                    rocblas_stride stride_alpha,
                                    const U __restrict__ Aa,
                                    rocblas_int strideA,
                                    const U __restrict__ xa,
                                    ptrdiff_t      shiftx,
                                    rocblas_int    incx,
                                    rocblas_int    stridex,
                                    V              beta_device_host,
                                    rocblas_stride stride_beta,
                                    U              ya,
                                    ptrdiff_t      shifty,
                                    rocblas_int    incy,
                                    rocblas_int    stridey)
{
    ptrdiff_t tx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(tx < n)
    {
        auto alpha              = load_scalar(alpha_device_host, hipBlockIdx_z, stride_alpha);
        auto beta               = load_scalar(beta_device_host, hipBlockIdx_z, stride_beta);
        const T* __restrict__ A = load_ptr_batch(Aa, hipBlockIdx_z, 0, strideA);
        const T* __restrict__ x = load_ptr_batch(xa, hipBlockIdx_z, shiftx, stridex);
        T* y                    = load_ptr_batch(ya, hipBlockIdx_z, shifty, stridey);

        T dot = 0.0;
        for(int i = 0; i < n; i++)
        {
            // TODO decode packing
            if(uplo == rocblas_fill_lower ? i <= tx : i >= tx)
                dot += x[i] * A[i * lda + tx];
        }

        y[tx] = beta * y[tx] + alpha * dot;
    }
}

template <typename U, typename V>
rocblas_status rocblas_spmv_template(rocblas_handle handle,
                                     rocblas_fill   uplo,
                                     rocblas_int    n,
                                     const U*       alpha,
                                     rocblas_stride stride_alpha,
                                     const V*       A,
                                     rocblas_stride strideA,
                                     const V*       x,
                                     rocblas_int    offsetx,
                                     rocblas_int    incx,
                                     rocblas_stride stridex,
                                     const U*       beta,
                                     rocblas_stride stride_beta,
                                     V*             y,
                                     rocblas_int    offsety,
                                     rocblas_int    incy,
                                     rocblas_stride stridey,
                                     rocblas_int    batch_count)
{
    //quick return
    if(!n || !batch_count)
        return rocblas_status_success;

    hipStream_t rocblas_stream = handle->rocblas_stream;

    // in case of negative inc shift pointer to end of data for negative indexing tid*inc
    auto shiftx = incx < 0 ? offsetx - ptrdiff_t(incx) * (n - 1) : offsetx;
    auto shifty = incy < 0 ? offsety - ptrdiff_t(incy) * (n - 1) : offsety;

    static constexpr int SPMV_DIM_X = 64;
    rocblas_int          blocks     = (n - 1) / SPMV_DIM_X + 1;

    dim3 spmv_grid(blocks, batch_count);
    dim3 spmv_threads(SPMV_DIM_X);

    if(handle->pointer_mode == rocblas_pointer_mode_device)
    {
        hipLaunchKernelGGL((spmv_kernel<U, V>),
                           spmv_grid,
                           spmv_threads,
                           0,
                           rocblas_stream,
                           uplo,
                           n,
                           alpha,
                           stride_alpha,
                           A,
                           strideA,
                           x,
                           shiftx,
                           incx,
                           stridex,
                           beta,
                           stride_beta,
                           y,
                           shifty,
                           incy,
                           stridey);
    }
    else
    {
        if(!*alpha && *beta == 1)
            return rocblas_status_success;

        hipLaunchKernelGGL((spmv_kernel<U, V>),
                           spmv_grid,
                           spmv_threads,
                           0,
                           rocblas_stream,
                           uplo,
                           n,
                           *alpha,
                           stride_alpha,
                           A,
                           strideA,
                           x,
                           shiftx,
                           incx,
                           stridex,
                           *beta,
                           stride_beta,
                           y,
                           shifty,
                           incy,
                           stridey);
    }

    return rocblas_status_success;
}

#endif
