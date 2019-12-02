/* ************************************************************************
 * Copyright 2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "handle.h"
#include "rocblas.h"

template <typename T, typename U, typename W>
__global__ void rocblas_symv_kernel(rocblas_fill   uplo,
                                    rocblas_int    n,
                                    V              alpha_device_host,
                                    rocblas_stride stride_alpha,
                                    const U __restrict__ Aa,
                                    ptrdiff_t   shiftA,
                                    rocblas_int lda,
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

template <typename T, typename U, typename V, typename W>
rocblas_status rocblas_symv_template(rocblas_handle handle,
                                     rocblas_fill   uplo,
                                     rocblas_int    n,
                                     const U*       alpha,
                                     rocblas_stride stride_alpha,
                                     const V*       A,
                                     rocblas_int    offseta,
                                     rocblas_int    lda,
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

    if(uplo == rocblas_fill_upper)
    {
        // SPMV_DIM_Y must be at least 4, 8 * 8 is very slow only 40Gflop/s
        static constexpr int SPMV_DIM_X = 64;
        static constexpr int SPMV_DIM_Y = 16;
        rocblas_int          blocks     = (m - 1) / (SPMV_DIM_X * 4) + 1;
        if(std::is_same<T, rocblas_double_complex>{})
            blocks = (m - 1) / (SPMV_DIM_X) + 1;
        dim3 spmv_grid(blocks, batch_count);
        dim3 spmv_threads(SPMV_DIM_X, SPMV_DIM_Y);

        if(handle->pointer_mode == rocblas_pointer_mode_device)
        {
            hipLaunchKernelGGL((spmv_kernel<SPMV_DIM_X, SPMV_DIM_Y, T>),
                               spmv_grid,
                               spmv_threads,
                               0,
                               rocblas_stream,
                               n,
                               alpha,
                               stride_alpha,
                               A,
                               offseta,
                               lda,
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

            hipLaunchKernelGGL((spmv_kernel<SPMV_DIM_X, SPMV_DIM_Y, T>),
                               spmv_grid,
                               spmv_threads,
                               0,
                               rocblas_stream,
                               n,
                               *alpha,
                               stride_alpha,
                               A,
                               offseta,
                               lda,
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
    }
    else // lower
    {
        // transpose
        // number of columns on the y-dim of the grid
        static constexpr int NB = 256;
        dim3                 spmv_grid(n, batch_count);
        dim3                 spmv_threads(NB);

        if(handle->pointer_mode == rocblas_pointer_mode_device)
        {
            hipLaunchKernelGGL((spmv_kernel<NB, T>),
                               spmv_grid,
                               spmv_threads,
                               0,
                               rocblas_stream,
                               m,
                               n,
                               alpha,
                               stride_alpha,
                               A,
                               offseta,
                               lda,
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

            hipLaunchKernelGGL((spmv_kernel<NB, T>),
                               spmv_grid,
                               spmv_threads,
                               0,
                               rocblas_stream,
                               n,
                               *alpha,
                               stride_alpha,
                               A,
                               offseta,
                               lda,
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
    }

    return rocblas_status_success;
}

#endif
