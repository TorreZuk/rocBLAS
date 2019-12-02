/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "rocblas_symv.hpp"
#include "handle.h"
#include "logging.h"
#include "rocblas.h"
#include "utility.h"
#include <limits>

namespace
{
    template <typename>
    constexpr char rocblas_symv_name[] = "unknown";
    template <>
    constexpr char rocblas_symv_name<float>[] = "rocblas_ssymv";
    template <>
    constexpr char rocblas_symv_name<double>[] = "rocblas_dsymv";

    template <typename T>
    rocblas_status rocblas_symv_impl(rocblas_handle handle,
                                     rocblas_fill   uplo,
                                     rocblas_int    n,
                                     const T*       alpha,
                                     const T*       A,
                                     rocblas_int    lda,
                                     const T*       x,
                                     rocblas_int    incx,
                                     const T*       beta,
                                     T*             y,
                                     rocblas_int    incy)
    {
        if(!handle)
            return rocblas_status_invalid_handle;
        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        auto layer_mode = handle->layer_mode;
        if(layer_mode
           & (rocblas_layer_mode_log_trace | rocblas_layer_mode_log_bench
              | rocblas_layer_mode_log_profile))
        {
            auto uplo_letter = rocblas_fill_letter(uplo);

            if(handle->pointer_mode == rocblas_pointer_mode_host)
            {
                if(layer_mode & rocblas_layer_mode_log_trace)
                    log_trace(handle,
                              rocblas_symv_name<T>,
                              uplo,
                              n,
                              log_trace_scalar_value(alpha),
                              A,
                              lda,
                              x,
                              incx,
                              log_trace_scalar_value(beta),
                              y,
                              incy);

                if(layer_mode & rocblas_layer_mode_log_bench)
                {
                    log_bench(handle,
                              "./rocblas-bench -f symv -r",
                              rocblas_precision_string<T>,
                              "--uplo",
                              uplo_letter,
                              "-n",
                              n,
                              LOG_BENCH_SCALAR_VALUE(alpha),
                              "--lda",
                              lda,
                              "--incx",
                              incx,
                              LOG_BENCH_SCALAR_VALUE(beta),
                              "--incy",
                              incy);
                }
            }
            else
            {
                if(layer_mode & rocblas_layer_mode_log_trace)
                    log_trace(handle,
                              rocblas_symv_name<T>,
                              uplo,
                              n,
                              alpha,
                              A,
                              lda,
                              x,
                              incx,
                              beta,
                              y,
                              incy);
            }

            if(layer_mode & rocblas_layer_mode_log_profile)
                log_profile(handle,
                            rocblas_symv_name<T>,
                            "uplo",
                            uplo_letter,
                            "N",
                            n,
                            "lda",
                            lda,
                            "incx",
                            incx,
                            "incy",
                            incy);
        }

        if(n < 0 || lda < n || lda < 1 || !incx || !incy)
            return rocblas_status_invalid_size;

        if(!n)
            return rocblas_status_success;

        if(!A || !x || !y || !alpha || !beta)
            return rocblas_status_invalid_pointer;

        return rocblas_symv_template<T>(
            handle, uplo, n, alpha, 0, A, 0, lda, 0, x, 0, incx, 0, beta, 0, y, 0, incy, 0, 1);
    }

} // namespace

/*
* ===========================================================================
*    C wrapper
* ===========================================================================
*/

extern "C" {

rocblas_status rocblas_ssymv(rocblas_handle handle,
                             rocblas_fill   uplo,
                             rocblas_int    n,
                             const float*   alpha,
                             const float*   A,
                             rocblas_int    lda,
                             const float*   x,
                             rocblas_int    incx,
                             const float*   beta,
                             float*         y,
                             rocblas_int    incy)
{
    return rocblas_symv_impl(handle, transA, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

rocblas_status rocblas_dsymv(rocblas_handle handle,
                             rocblas_fill   uplo,
                             rocblas_int    n,
                             const double*  alpha,
                             const double*  A,
                             rocblas_int    lda,
                             const double*  x,
                             rocblas_int    incx,
                             const double*  beta,
                             double*        y,
                             rocblas_int    incy)
{
    return rocblas_symv_impl(handle, transA, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

} // extern "C"
