/* ************************************************************************
 * Copyright 2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#ifndef __ROCBLAS_SYMM_HPP__
#define __ROCBLAS_SYMM_HPP__

#include "handle.h"
#include "rocblas.h"
#include "utility.h"

template <typename T>
static __device__ void
    symm_scale_device(rocblas_int m, rocblas_int n, T beta, T* C, rocblas_int ldc)
{
    auto tx = blockIdx.x * blockDim.x + threadIdx.x;
    auto ty = blockIdx.y * blockDim.y + threadIdx.y;

    if(tx < m && ty < n)
    {
        C[ty * ldc + tx] *= beta;
    }
}

/**
  *  Loads pointers and launches the actual calculation kernel.
  */
template <typename T, typename U>
__global__ void symm_scale_kernel(rocblas_int    m,
                                  rocblas_int    n,
                                  T              beta_host_device,
                                  U              CP_array,
                                  ptrdiff_t      shift_c,
                                  rocblas_int    ldc,
                                  rocblas_stride stride_c)
{
    auto beta = load_scalar(beta_host_device);
    if(beta == 1)
        return;

    auto C = load_ptr_batch(CP_array, hipBlockIdx_z, shift_c, stride_c);
    symm_scale_device(m, n, beta, C, ldc);
}

/**
  * kernel
  */
template <bool HERM, bool RIGHT, rocblas_int TILE_NK, typename T>
static __device__ void symm_hemm_mult_add_device(bool        upper,
                                                 rocblas_int m,
                                                 rocblas_int n,
                                                 T           alpha,
                                                 const T* __restrict__ A,
                                                 rocblas_int lda,
                                                 const T* __restrict__ B,
                                                 rocblas_int ldb,
                                                 T* __restrict__ C,
                                                 rocblas_int ldc)
{
    __shared__ T atile[TILE_NK][TILE_NK];
    __shared__ T btile[TILE_NK][TILE_NK];

    int col_pos = blockIdx.y * TILE_NK;
    int row_pos = blockIdx.x * TILE_NK;

    int tilefrom = upper ? row_pos : col_pos;
    int tileto   = upper ? col_pos : row_pos;
    if(tilefrom > tileto)
    {
        // any overlap of tile and output
        return;
    }

    int ab_rows = !RIGHT ? m : n;
    int ab_cols = !RIGHT ? n : m;

    int row = row_pos + threadIdx.x;
    int col = col_pos + threadIdx.y;

    int from = upper ? row : col;
    int to   = upper ? col : row;

    for(int k_pos = 0; k_pos < n; k_pos += TILE_NK)
    {
        // tiling over dimension K

        int row_loc, col_loc, k_loc;
        int r, c;

        // first matrix mult: alpha*op(A)*op(B)^T
        // when HERM ^H instead of ^T

        // fetch tile of matrix A
        row_loc = row_pos + threadIdx.x;
        col_loc = k_pos + threadIdx.y;
        r       = RIGHT ? col_loc : row_loc; // RIGHT A = A^T, else A = A
        c       = RIGHT ? row_loc : col_loc;

        atile[threadIdx.x][threadIdx.y]
            = (r < ab_rows && c < ab_cols) ? (HERM && RIGHT ? conj(A[c * lda + r]) : A[c * lda + r])
                                           : 0;

        // fetch tile of matrix B
        row_loc = k_pos + threadIdx.x;
        col_loc = col_pos + threadIdx.y;
        r       = RIGHT ? row_loc : col_loc; // RIGHT B = B, else B = B^T
        c       = RIGHT ? col_loc : row_loc;

        btile[threadIdx.x][threadIdx.y]
            = (c < ab_cols && r < ab_rows)
                  ? (HERM && !RIGHT ? conj(B[c * ldb + r]) : B[c * ldb + r])
                  : 0;

        __syncthreads();

        // m x m symmetric/hermitian output, tile zero where invalid
        if(row < m && col < m && from <= to)
        {
            T sum = T(0);
            for(int ki = 0; ki < TILE_NK; ++ki)
            {
                sum += atile[threadIdx.x][ki] * btile[ki][threadIdx.y];
            }
            C[col * ldc + row] += alpha * sum;
        }

        __syncthreads();

        // second matrix mult: alpha*op(B)*op(A)^T, if HERM conj(alpha) and ^H
        if(HERM)
        {
            // fetch tile of matrix B  into tileA
            row_loc = row_pos + threadIdx.x;
            col_loc = k_pos + threadIdx.y;
            r       = RIGHT ? col_loc : row_loc; // RIGHT B = B^T, else B = B
            c       = RIGHT ? row_loc : col_loc;

            atile[threadIdx.x][threadIdx.y]
                = (r < ab_rows && c < ab_cols)
                      ? (HERM && RIGHT ? conj(B[c * ldb + r]) : B[c * ldb + r])
                      : 0;

            // fetch tile of matrix A into tileB
            row_loc = k_pos + threadIdx.x;
            col_loc = col_pos + threadIdx.y;
            r       = RIGHT ? row_loc : col_loc; // RIGHT A = A, else A = A^T
            c       = RIGHT ? col_loc : row_loc;

            btile[threadIdx.x][threadIdx.y]
                = (c < ab_cols && r < ab_rows)
                      ? (HERM && !RIGHT ? conj(A[c * lda + r]) : A[c * lda + r])
                      : 0;

            __syncthreads();

            // m x m symmetric/hermitian output, tile zero where invalid
            if(row < m && col < m && from <= to)
            {
                T sum = T(0);
                for(int ki = 0; ki < TILE_NK; ++ki)
                {
                    sum += atile[threadIdx.x][ki] * btile[ki][threadIdx.y];
                }
                C[col * ldc + row] += (HERM ? conj(alpha) : alpha) * sum;
            }

            __syncthreads();
        }

    } // k_pos

    // if(HERM && row == col && row < m)
    // {
    //     // zero imaginary in case of numerical drift
    //     C[col * ldc + row].y = 0;
    // }
}

/**
  *  Loads pointers and launches the actual calculation kernel.
  */
template <bool        HERM,
          bool        RIGHT,
          rocblas_int DIM_XYT,
          typename TScal,
          typename TConstPtr,
          typename TPtr>
__global__ void symm_hemm_kernel(bool           upper,
                                 rocblas_int    m,
                                 rocblas_int    n,
                                 TScal          alpha_host_device,
                                 TConstPtr      AP_array,
                                 ptrdiff_t      shift_a,
                                 rocblas_int    lda,
                                 rocblas_stride stride_a,
                                 TConstPtr      BP_array,
                                 ptrdiff_t      shift_b,
                                 rocblas_int    ldb,
                                 rocblas_stride stride_b,
                                 TPtr           CP_array,
                                 ptrdiff_t      shift_c,
                                 rocblas_int    ldc,
                                 rocblas_stride stride_c)
{
    auto alpha = load_scalar(alpha_host_device);
    if(alpha == 0)
        return;

    auto A = load_ptr_batch(AP_array, hipBlockIdx_z, shift_a, stride_a);
    auto B = load_ptr_batch(BP_array, hipBlockIdx_z, shift_b, stride_b);
    auto C = load_ptr_batch(CP_array, hipBlockIdx_z, shift_c, stride_c);

    // compute matrix multiplies and accumulate on the fly into C
    // when HERM does ^H in place of ^T
    symm_hemm_mult_add_device<HERM, RIGHT, DIM_XYT>(upper, m, n, alpha, A, lda, B, ldb, C, ldc);
}

template <typename TScal, typename TConstPtr, typename TPtr>
rocblas_status rocblas_symm_arg_check(rocblas_handle handle,
                                      rocblas_side   side,
                                      rocblas_fill   uplo,
                                      rocblas_int    m,
                                      rocblas_int    n,
                                      TScal          alpha,
                                      TConstPtr      AP,
                                      rocblas_int    offsetA,
                                      rocblas_int    lda,
                                      rocblas_stride strideA,
                                      TConstPtr      BP,
                                      rocblas_int    offsetB,
                                      rocblas_int    ldb,
                                      rocblas_stride strideB,
                                      TScal          beta,
                                      TPtr           CP,
                                      rocblas_int    offsetC,
                                      rocblas_int    ldc,
                                      rocblas_stride strideC,
                                      rocblas_int    batch_count)
{

    if(side != rocblas_side_left && side != rocblas_side_right)
        return rocblas_status_invalid_value;

    if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
        return rocblas_status_invalid_value;

    if(batch_count < 0 || m < 0 || n < 0 || ldc < m || ldb < m
       || (side == rocblas_side_left && (lda < m)) || (side != rocblas_side_left && (lda < n)))
        return rocblas_status_invalid_size;

    if(!m || !n || !batch_count)
        return rocblas_status_success;

    if((n > 0 && (!AP || !BP || !alpha)) || !CP || !beta)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}
/**
  *  TScal     is always: const T* (either host or device)
  *  TConstPtr is either: const T* OR const T* const*
  *  TPtr      is either:       T* OR       T* const*
  */
template <bool HERM, typename TScal, typename TConstPtr, typename TPtr>
rocblas_status rocblas_symm_template(rocblas_handle handle,
                                     rocblas_side   side,
                                     rocblas_fill   uplo,
                                     rocblas_int    m,
                                     rocblas_int    n,
                                     TScal          alpha,
                                     TConstPtr      AP,
                                     rocblas_int    offsetA,
                                     rocblas_int    lda,
                                     rocblas_stride strideA,
                                     TConstPtr      BP,
                                     rocblas_int    offsetB,
                                     rocblas_int    ldb,
                                     rocblas_stride strideB,
                                     TScal          beta,
                                     TPtr           CP,
                                     rocblas_int    offsetC,
                                     rocblas_int    ldc,
                                     rocblas_stride strideC,
                                     rocblas_int    batch_count)
{
    // quick return
    if(!m || !batch_count)
        return rocblas_status_success;

    static constexpr int symm_SCALE_DIM_X = 128;
    static constexpr int symm_SCALE_DIM_Y = 8;
    rocblas_int          gx               = (m - 1) / (symm_SCALE_DIM_X) + 1;
    rocblas_int          gy               = (m - 1) / (symm_SCALE_DIM_Y) + 1;
    dim3                 symm_scale_grid(gx, gy, batch_count);
    dim3                 symm_scale_threads(symm_SCALE_DIM_X, symm_SCALE_DIM_Y);

    static constexpr int symm_DIM_XY = 32;
    rocblas_int          bx          = (m - 1) / (symm_DIM_XY) + 1;
    rocblas_int          by          = (m - 1) / (symm_DIM_XY) + 1;
    dim3                 symm_grid(bx, by, batch_count);
    dim3                 symm_threads(symm_DIM_XY, symm_DIM_XY);

    // Launch a herk kernel for symm.
    if(handle->pointer_mode == rocblas_pointer_mode_device)
    {
        // first scale C so we can use directly for output without work buffer
        hipLaunchKernelGGL((symm_scale_kernel),
                           symm_scale_grid,
                           symm_scale_threads,
                           0,
                           handle->rocblas_stream,
                           m,
                           n,
                           beta,
                           CP,
                           offsetC,
                           ldc,
                           strideC);

        if(m == 0 || n == 0)
            return rocblas_status_success;

        if(side == rocblas_side_left)
        {
            hipLaunchKernelGGL((symm_hemm_kernel<HERM, false, symm_DIM_XY>),
                               symm_grid,
                               symm_threads,
                               0,
                               handle->rocblas_stream,
                               uplo == rocblas_fill_upper,
                               m,
                               n,
                               alpha,
                               AP,
                               offsetA,
                               lda,
                               strideA,
                               BP,
                               offsetB,
                               ldb,
                               strideB,
                               CP,
                               offsetC,
                               ldc,
                               strideC);
        }
        else
        {
            hipLaunchKernelGGL((symm_hemm_kernel<HERM, true, symm_DIM_XY>),
                               symm_grid,
                               symm_threads,
                               0,
                               handle->rocblas_stream,
                               uplo == rocblas_fill_upper,
                               m,
                               n,
                               alpha,
                               AP,
                               offsetA,
                               lda,
                               strideA,
                               BP,
                               offsetB,
                               ldb,
                               strideB,
                               CP,
                               offsetC,
                               ldc,
                               strideC);
        }
    }
    else
    {
        if(*beta == 1 && (*alpha == 0 || m == 0 || n == 0))
            return rocblas_status_success;

        // first scale C so we can use directly for output without work buffer
        hipLaunchKernelGGL((symm_scale_kernel),
                           symm_scale_grid,
                           symm_scale_threads,
                           0,
                           handle->rocblas_stream,
                           m,
                           n,
                           *beta,
                           CP,
                           offsetC,
                           ldc,
                           strideC);

        if(n == 0)
            return rocblas_status_success;

        if(side == rocblas_side_left)
        {
            hipLaunchKernelGGL((symm_hemm_kernel<HERM, false, symm_DIM_XY>),
                               symm_grid,
                               symm_threads,
                               0,
                               handle->rocblas_stream,
                               uplo == rocblas_fill_upper,
                               m,
                               n,
                               *alpha,
                               AP,
                               offsetA,
                               lda,
                               strideA,
                               BP,
                               offsetB,
                               ldb,
                               strideB,
                               CP,
                               offsetC,
                               ldc,
                               strideC);
        }
        else
        {
            hipLaunchKernelGGL((symm_hemm_kernel<HERM, true, symm_DIM_XY>),
                               symm_grid,
                               symm_threads,
                               0,
                               handle->rocblas_stream,
                               uplo == rocblas_fill_upper,
                               m,
                               n,
                               *alpha,
                               AP,
                               offsetA,
                               lda,
                               strideA,
                               BP,
                               offsetB,
                               ldb,
                               strideB,
                               CP,
                               offsetC,
                               ldc,
                               strideC);
        }
    }

    return rocblas_status_success;
}

#endif
