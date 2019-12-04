/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#ifndef _ROCLAPACK_FUNCTIONS_H
#define _ROCLAPACK_FUNCTIONS_H

#include "rocblas-types.h"
#include "rocsolver-export.h"

/*! \file
    \brief rocsolver_functions.h provides Lapack functionality for the ROCm platform.
 *********************************************************************************/

/*
 * ===========================================================================
 *   READEME: Please follow the naming convention
 *   Big case for matrix, e.g. matrix A, B, C   GEMM (C = A*B)
 *   Lower case for vector, e.g. vector x, y    GEMV (y = A*x)
 * ===========================================================================
 */

#ifdef __cplusplus
extern "C" {
#endif

/*
 * ===========================================================================
 *      Auxiliary functions
 * ===========================================================================
 */

/*! \brief LASWP performs a series of row interchanges on the matrix A.

    \details
    It interchanges row I with row IPIV[k1 + (I - k1) * abs(inx)], for
    each of rows K1 through K2 of A. k1 and k2 are 1-based indices.

    @param[in]
    handle          rocblas_handle
    @param[in]
    n               rocblas_int. n >= 0.\n
                    The number of columns of the matrix A.
    @param[inout]
    A               pointer to type. Array on the GPU of dimension lda*n. \n
                    On entry, the matrix of column dimension n to which the row
                    interchanges will be applied. On exit, the permuted matrix.
    @param[in]
    lda             rocblas_int. lda > 0.\n
                    The leading dimension of the array A.
    @param[in]
    k1              rocblas_int. k1 > 0.\n
                    The first element of IPIV for which a row interchange will
                    be done. This is a 1-based index.
    @param[in]
    k2              rocblas_int. k2 > k1 > 0.\n
                    (K2-K1+1) is the number of elements of IPIV for which a row
                    interchange will be done. This is a 1-based index.
    @param[in]
    ipiv            pointer to rocblas_int. Array on the GPU of dimension at least k1 + (k2 - k1) * abs(incx).\n
                    The vector of pivot indices.  Only the elements in positions
                    k1 through (k1 + (k2 - k1) * abs(incx)) of IPIV are accessed.
                    Elements of ipiv are considered 1-based.
    @param[in]
    incx            rocblas_int. incx != 0.\n
                    The increment between successive values of IPIV.  If IPIV
                    is negative, the pivots are applied in reverse order.
    *************************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_slaswp(rocblas_handle     handle,
                                                 const rocblas_int  n,
                                                 float*             A,
                                                 const rocblas_int  lda,
                                                 const rocblas_int  k1,
                                                 const rocblas_int  k2,
                                                 const rocblas_int* ipiv,
                                                 const rocblas_int  incx);

ROCSOLVER_EXPORT rocblas_status rocsolver_dlaswp(rocblas_handle     handle,
                                                 const rocblas_int  n,
                                                 double*            A,
                                                 const rocblas_int  lda,
                                                 const rocblas_int  k1,
                                                 const rocblas_int  k2,
                                                 const rocblas_int* ipiv,
                                                 const rocblas_int  incx);

/*! \brief LARFG generates an orthogonal Householder reflector H of order n.

    \details
    Householder reflector H is such that

        H * [alpha] = [beta]
            [  x  ]   [  0 ]

    where x is an n-1 vector and alpha and beta are scalars. Matrix H can be
    generated as

        H = I - tau * [1] * [1 v']
                      [v]

    with v an n-1 vector and tau a scalar.

    @param[in]
    handle          rocblas_handle
    @param[in]
    n               rocblas_int. n >= 0.\n
                    The order (size) of reflector H.
    @param[inout]
    alpha           pointer to type. A scalar on the GPU.\n
                    On input the scalar alpha,
                    on output it is overwritten with beta.
    @param[inout]
    x               pointer to type. Array on the GPU of size at least n-1.\n
                    On input it is the vector x,
                    on output it is overwritten with vector v.
    @param[in]
    incx            rocblas_int. incx > 0.\n
                    The increment between consecutive elements of x.
    @param[out]
    tau             pointer to type. A scalar on the GPU.\n
                    The scalar tau.

    *************************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_slarfg(rocblas_handle    handle,
                                                 const rocblas_int n,
                                                 float*            alpha,
                                                 float*            x,
                                                 const rocblas_int incx,
                                                 float*            tau);

ROCSOLVER_EXPORT rocblas_status rocsolver_dlarfg(rocblas_handle    handle,
                                                 const rocblas_int n,
                                                 double*           alpha,
                                                 double*           x,
                                                 const rocblas_int incx,
                                                 double*           tau);

/*! \brief LARFT Generates the triangular factor T of a block reflector H of order n.

    \details
    The block reflector H is defined as the product of k Householder matrices as

        H = H(1) * H(2) * ... * H(k)  (forward direction), or
        H = H(k) * ... * H(2) * H(1)  (backward direction)

    depending on the value of direct.

    The triangular matrix T is such that

        H = I - V * T * V'

    where the i-th column of matrix V contains the Householder vector associated to H(i).

    @param[in]
    handle              rocblas_handle.
    @param[in]
    direct              rocblas_direct.\n
                        Specifies the direction in which the Householder matrices are applied.
    @param[in]
    n                   rocblas_int. n >= 0.\n
                        The order (size) of the block reflector.
    @param[in]
    k                   rocsovler_int. k >= 1.\n
                        The number of Householder matrices.
    @param[in]
    V                   pointer to type. Array on the GPU of size ldv*k.\n
                        The matrix of Householder vectors.
    @param[in]
    ldv                 rocblas_int. ldv >= n.\n
                        Leading dimension of V.
    @param[in]
    tau                 pointer to type. Array of k scalars on the GPU.\n
                        The vector of all the scalars associated to the Householder matrices.
    @param[out]
    T                   pointer to type. Array on the GPU of dimension ldt*k.\n
                        The triangular factor. T is upper triangular is forward operation, otherwise it is lower triangular.
                        The rest of the array is not used.
    @param[in]
    ldt                 rocblas_int. ldt >= k.\n
                        The leading dimension of T.

    **************************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_slarft(rocblas_handle       handle,
                                                 const rocblas_direct direct,
                                                 const rocblas_int    n,
                                                 const rocblas_int    k,
                                                 float*               V,
                                                 const rocblas_int    ldv,
                                                 float*               tau,
                                                 float*               T,
                                                 const rocblas_int    ldt);

ROCSOLVER_EXPORT rocblas_status rocsolver_dlarft(rocblas_handle       handle,
                                                 const rocblas_direct direct,
                                                 const rocblas_int    n,
                                                 const rocblas_int    k,
                                                 double*              V,
                                                 const rocblas_int    ldv,
                                                 double*              tau,
                                                 double*              T,
                                                 const rocblas_int    ldt);

/*! \brief LARF applies a Householder reflector H to a general matrix A.

    \details
    The Householder reflector H, of order m (or n), is to be applied to a m-by-n matrix A
    from the left (or the right). H is given by

        H = I - alpha * x * x'

    where alpha is a scalar and x a Householder vector. H is never actually computed.

    @param[in]
    handle          rocblas_handle.
    @param[in]
    side            rocblas_side.\n
                    If side = rocsolver_side_left, then compute H*A
                    If side = rocsolver_side_right, then compute A*H
    @param[in]
    m               rocblas_int. m >= 0.\n
                    Number of rows of A.
    @param[in]
    n               rocblas_int. n >= 0.\n
                    Number of columns of A.
    @param[in]
    x               pointer to type. Array on the GPU of
                    size at least (1 + (m-1)*abs(incx)) if left side, or
                    at least (1 + (n-1)*abs(incx)) if right side.\n
                    The Householder vector x.
    @param[in]
    incx            rocblas_int. incx != 0.\n
                    Increment between to consecutive elements of x.
                    If incx < 0, the elements of x are used in reverse order.
    @param[in]
    alpha           pointer to type. A scalar on the GPU.\n
                    If alpha = 0, then H = I (A will remain the same, x is never used)
    @param[inout]
    A               pointer to type. Array on the GPU of size lda*n.\n
                    On input, the matrix A. On output it is overwritten with
                    H*A (or A*H).
    @param[in]
    lda             rocblas_int. lda >= m.\n
                    Leading dimension of A.

    *************************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_slarf(rocblas_handle     handle,
                                                const rocblas_side side,
                                                const rocblas_int  m,
                                                const rocblas_int  n,
                                                float*             x,
                                                const rocblas_int  incx,
                                                const float*       alpha,
                                                float*             A,
                                                const rocblas_int  lda);

ROCSOLVER_EXPORT rocblas_status rocsolver_dlarf(rocblas_handle     handle,
                                                const rocblas_side side,
                                                const rocblas_int  m,
                                                const rocblas_int  n,
                                                double*            x,
                                                const rocblas_int  incx,
                                                const double*      alpha,
                                                double*            A,
                                                const rocblas_int  lda);

/*! \brief LARFB applies a block reflector H to a general m-by-n matrix A.

    \details
    The block reflector H is applied in one of the following forms, depending on
    the values of side and trans:

        H  * A  (No transpose from the left)
        H' * A  (Transpose from the left)
        A * H   (No transpose from the right), and
        A * H'  (Transpose from the right)

    The block reflector H is defined as the product of k Householder matrices as

        H = H(1) * H(2) * ... * H(k)  (forward direction), or
        H = H(k) * ... * H(2) * H(1)  (backward direction)

    depending on the value of direct. H is never stored. It is calculated as

        H = I - V * T * V'

    where the i-th column of matrix V contains the Householder vector associated to H(i),
    and T is the triangular factor as computed by LARFT.

    @param[in]
    handle              rocblas_handle.
    @param[in]
    side                rocblas_side.\n
                        Specifies from which side to apply H.
    @param[in]
    trans               rocblas_operation.\n
                        Specifies whether the block reflector or its transpose is to be applied.
    @param[in]
    direct              rocblas_direct.\n
                        Specifies the direction in which the Householder matrices are applied.
    @param[in]
    m                   rocblas_int. m >= 0.\n
                        Number of rows of matrix A.
    @param[in]
    n                   rocblas_int. n >= 0.\n
                        Number of columns of matrix A.
    @param[in]
    k                   rocsovler_int. k >= 1.\n
                        The number of Householder matrices.
    @param[in]
    V                   pointer to type. Array on the GPU of size ldv*k.\n
                        The matrix of Householder vectors.
    @param[in]
    ldv                 rocblas_int. ldv >= m if side is left, or ldv >= n if side is right.\n
                        Leading dimension of V.
    @param[in]
    T                   pointer to type. Array on the GPU of dimension ldt*k.\n
                        The triangular factor of the block reflector.
    @param[in]
    ldt                 rocblas_int. ldt >= k.\n
                        The leading dimension of T.
    @param[inout]
    A                   pointer to type. Array on the GPU of size lda*n.\n
                        On input, the matrix A. On output it is overwritten with
                        H*A, A*H, H'*A, or A*H'.
    @param[in]
    lda                 rocblas_int. lda >= m.\n
                        Leading dimension of A.

    ****************************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_slarfb(rocblas_handle          handle,
                                                 const rocblas_side      side,
                                                 const rocblas_operation trans,
                                                 const rocblas_direct    direct,
                                                 const rocblas_int       m,
                                                 const rocblas_int       n,
                                                 const rocblas_int       k,
                                                 float*                  V,
                                                 const rocblas_int       ldv,
                                                 float*                  T,
                                                 const rocblas_int       ldt,
                                                 float*                  A,
                                                 const rocblas_int       lda);

ROCSOLVER_EXPORT rocblas_status rocsolver_dlarfb(rocblas_handle          handle,
                                                 const rocblas_side      side,
                                                 const rocblas_operation trans,
                                                 const rocblas_direct    direct,
                                                 const rocblas_int       m,
                                                 const rocblas_int       n,
                                                 const rocblas_int       k,
                                                 double*                 V,
                                                 const rocblas_int       ldv,
                                                 double*                 T,
                                                 const rocblas_int       ldt,
                                                 double*                 A,
                                                 const rocblas_int       lda);

/*
 * ===========================================================================
 *      LAPACK functions
 * ===========================================================================
 */

/*! \brief GETF2 computes the LU factorization of a general m-by-n matrix A
    using partial pivoting with row interchanges.

    \details
    (This is the right-looking Level 2 BLAS version of the algorithm).

    The factorization has the form

        A = P * L * U

    where P is a permutation matrix, L is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U is upper
    triangular (upper trapezoidal if m < n).

    @param[in]
    handle    rocblas_handle.
    @param[in]
    m         rocblas_int. m >= 0.\n
              The number of rows of the matrix A.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The number of colums of the matrix A.
    @param[inout]
    A         pointer to type. Array on the GPU of dimension lda*n.\n
              On entry, the m-by-n matrix A to be factored.
              On exit, the factors L and U from the factorization.
              The unit diagonal elements of L are not stored.
    @param[in]
    lda       rocblas_int. lda >= m.\n
              Specifies the leading dimension of A.
    @param[out]
    ipiv      pointer to rocblas_int. Array on the GPU of dimension min(m,n).\n
              The vector of pivot indices. Elements of ipiv are 1-based indices.
              For 1 <= i <= min(m,n), the row i of the
              matrix was interchanged with row ipiv[i].
              Matrix P of the factorization can be derived from ipiv.
    @param[out]
    info      pointer to a rocblas_int on the GPU.\n
              If info = 0, succesful exit.
              If info = i > 0, U is singular. U(i,i) is the first zero pivot.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgetf2(rocblas_handle    handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 float*            A,
                                                 const rocblas_int lda,
                                                 rocblas_int*      ipiv,
                                                 rocblas_int*      info);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgetf2(rocblas_handle    handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 double*           A,
                                                 const rocblas_int lda,
                                                 rocblas_int*      ipiv,
                                                 rocblas_int*      info);

/*! \brief GETF2_BATCHED computes the LU factorization of a batch of general m-by-n matrices
    using partial pivoting with row interchanges.

    \details
    (This is the right-looking Level 2 BLAS version of the algorithm).

    The factorization of matrix A_i in the batch has the form

        A_i = P_i * L_i * U_i

    where P_i is a permutation matrix, L_i is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U_i is upper
    triangular (upper trapezoidal if m < n).

    @param[in]
    handle    rocblas_handle.
    @param[in]
    m         rocblas_int. m >= 0.\n
              The number of rows of all matrices A_i in the batch.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The number of colums of all matrices A_i in the batch.
    @param[inout]
    A         array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
              On entry, the m-by-n matrices A_i to be factored.
              On exit, the factors L_i and U_i from the factorizations.
              The unit diagonal elements of L_i are not stored.
    @param[in]
    lda       rocblas_int. lda >= m.\n
              Specifies the leading dimension of matrices A_i.
    @param[out]
    ipiv      pointer to rocblas_int. Array on the GPU (the size depends on the value of strideP).\n
              Contains the vectors of pivot indices ipiv_i (corresponding to A_i).
              Dimension of ipiv_i is min(m,n).
              Elements of ipiv_i are 1-based indices.
              For each instance A_i in the batch and for 1 <= j <= min(m,n), the row j of the
              matrix A_i was interchanged with row ipiv_i[j].
              Matrix P_i of the factorization can be derived from ipiv_i.
    @param[in]
    strideP   rocblas_int.\n
              Stride from the start of one vector ipiv_i to the next one ipiv_(i+1).
              There is no restriction for the value of strideP. Normal use case is strideP >= min(m,n).
    @param[out]
    info      pointer to rocblas_int. Array of batch_count integers on the GPU.\n
              If info_i = 0, succesful exit for factorization of A_i.
              If info_i = j > 0, U_i is singular. U_i(j,j) is the first zero pivot.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
                Number of matrices in the batch.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgetf2_batched(rocblas_handle    handle,
                                                         const rocblas_int m,
                                                         const rocblas_int n,
                                                         float* const      A[],
                                                         const rocblas_int lda,
                                                         rocblas_int*      ipiv,
                                                         const rocblas_int strideP,
                                                         rocblas_int*      info,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgetf2_batched(rocblas_handle    handle,
                                                         const rocblas_int m,
                                                         const rocblas_int n,
                                                         double* const     A[],
                                                         const rocblas_int lda,
                                                         rocblas_int*      ipiv,
                                                         const rocblas_int strideP,
                                                         rocblas_int*      info,
                                                         const rocblas_int batch_count);

/*! \brief GETF2_STRIDED_BATCHED computes the LU factorization of a batch of general m-by-n matrices
    using partial pivoting with row interchanges.

    \details
    (This is the right-looking Level 2 BLAS version of the algorithm).

    The factorization of matrix A_i in the batch has the form

        A_i = P_i * L_i * U_i

    where P_i is a permutation matrix, L_i is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U_i is upper
    triangular (upper trapezoidal if m < n).

    @param[in]
    handle    rocblas_handle.
    @param[in]
    m         rocblas_int. m >= 0.\n
              The number of rows of all matrices A_i in the batch.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The number of colums of all matrices A_i in the batch.
    @param[inout]
    A         pointer to type. Array on the GPU (the size depends on the value of strideA).\n
              On entry, in contains the m-by-n matrices A_i to be factored.
              On exit, the factors L_i and U_i from the factorization.
              The unit diagonal elements of L_i are not stored.
    @param[in]
    lda       rocblas_int. lda >= m.\n
              Specifies the leading dimension of matrices A_i.
    @param[in]
    strideA   rocblas_int.\n
              Stride from the start of one matrix A_i and the next one A_(i+1).
              There is no restriction for the value of strideA. Normal use case is strideA >= lda*n
    @param[out]
    ipiv      pointer to rocblas_int. Array on the GPU (the size depends on the value of strideP).\n
              Contains the vectors of pivots indices ipiv_i (corresponding to A_i).
              Dimension of ipiv_i is min(m,n).
              Elements of ipiv_i are 1-based indices.
              For each instance A_i in the batch and for 1 <= j <= min(m,n), the row j of the
              matrix A_i was interchanged with row ipiv_i[j].
              Matrix P_i of the factorization can be derived from ipiv_i.
    @param[in]
    strideP   rocblas_int.\n
              Stride from the start of one vector ipiv_i to the next one ipiv_(i+1).
              There is no restriction for the value of strideP. Normal use case is strideP >= min(m,n).
    @param[out]
    info      pointer to rocblas_int. Array of batch_count integers on the GPU.\n
              If info_i = 0, succesful exit for factorization of A_i.
              If info_i = j > 0, U_i is singular. U_i(j,j) is the first zero pivot.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
                Number of matrices in the batch.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgetf2_strided_batched(rocblas_handle    handle,
                                                                 const rocblas_int m,
                                                                 const rocblas_int n,
                                                                 float*            A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_int strideA,
                                                                 rocblas_int*      ipiv,
                                                                 const rocblas_int strideP,
                                                                 rocblas_int*      info,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgetf2_strided_batched(rocblas_handle    handle,
                                                                 const rocblas_int m,
                                                                 const rocblas_int n,
                                                                 double*           A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_int strideA,
                                                                 rocblas_int*      ipiv,
                                                                 const rocblas_int strideP,
                                                                 rocblas_int*      info,
                                                                 const rocblas_int batch_count);

/*! \brief GETRF computes the LU factorization of a general m-by-n matrix A
    using partial pivoting with row interchanges.

    \details
    (This is the right-looking Level 3 BLAS version of the algorithm).

    The factorization has the form

        A = P * L * U

    where P is a permutation matrix, L is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U is upper
    triangular (upper trapezoidal if m < n).

    @param[in]
    handle    rocblas_handle.
    @param[in]
    m         rocblas_int. m >= 0.\n
              The number of rows of the matrix A.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The number of colums of the matrix A.
    @param[inout]
    A         pointer to type. Array on the GPU of dimension lda*n.\n
              On entry, the m-by-n matrix A to be factored.
              On exit, the factors L and U from the factorization.
              The unit diagonal elements of L are not stored.
    @param[in]
    lda       rocblas_int. lda >= m.\n
              Specifies the leading dimension of A.
    @param[out]
    ipiv      pointer to rocblas_int. Array on the GPU of dimension min(m,n).\n
              The vector of pivot indices. Elements of ipiv are 1-based indices.
              For 1 <= i <= min(m,n), the row i of the
              matrix was interchanged with row ipiv[i].
              Matrix P of the factorization can be derived from ipiv.
    @param[out]
    info      pointer to a rocblas_int on the GPU.\n
              If info = 0, succesful exit.
              If info = i > 0, U is singular. U(i,i) is the first zero pivot.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgetrf(rocblas_handle    handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 float*            A,
                                                 const rocblas_int lda,
                                                 rocblas_int*      ipiv,
                                                 rocblas_int*      info);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgetrf(rocblas_handle    handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 double*           A,
                                                 const rocblas_int lda,
                                                 rocblas_int*      ipiv,
                                                 rocblas_int*      info);

/*! \brief GETRF_BATCHED computes the LU factorization of a batch of general m-by-n matrices
    using partial pivoting with row interchanges.

    \details
    (This is the right-looking Level 3 BLAS version of the algorithm).

    The factorization of matrix A_i in the batch has the form

        A_i = P_i * L_i * U_i

    where P_i is a permutation matrix, L_i is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U_i is upper
    triangular (upper trapezoidal if m < n).

    @param[in]
    handle    rocblas_handle.
    @param[in]
    m         rocblas_int. m >= 0.\n
              The number of rows of all matrices A_i in the batch.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The number of colums of all matrices A_i in the batch.
    @param[inout]
    A         array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
              On entry, the m-by-n matrices A_i to be factored.
              On exit, the factors L_i and U_i from the factorizations.
              The unit diagonal elements of L_i are not stored.
    @param[in]
    lda       rocblas_int. lda >= m.\n
              Specifies the leading dimension of matrices A_i.
    @param[out]
    ipiv      pointer to rocblas_int. Array on the GPU (the size depends on the value of strideP).\n
              Contains the vectors of pivot indices ipiv_i (corresponding to A_i).
              Dimension of ipiv_i is min(m,n).
              Elements of ipiv_i are 1-based indices.
              For each instance A_i in the batch and for 1 <= j <= min(m,n), the row j of the
              matrix A_i was interchanged with row ipiv_i(j).
              Matrix P_i of the factorization can be derived from ipiv_i.
    @param[in]
    strideP   rocblas_int.\n
              Stride from the start of one vector ipiv_i to the next one ipiv_(i+1).
              There is no restriction for the value of strideP. Normal use case is strideP >= min(m,n).
    @param[out]
    info      pointer to rocblas_int. Array of batch_count integers on the GPU.\n
              If info_i = 0, succesful exit for factorization of A_i.
              If info_i = j > 0, U_i is singular. U_i(j,j) is the first zero pivot.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
                Number of matrices in the batch.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgetrf_batched(rocblas_handle    handle,
                                                         const rocblas_int m,
                                                         const rocblas_int n,
                                                         float* const      A[],
                                                         const rocblas_int lda,
                                                         rocblas_int*      ipiv,
                                                         const rocblas_int strideP,
                                                         rocblas_int*      info,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgetrf_batched(rocblas_handle    handle,
                                                         const rocblas_int m,
                                                         const rocblas_int n,
                                                         double* const     A[],
                                                         const rocblas_int lda,
                                                         rocblas_int*      ipiv,
                                                         const rocblas_int strideP,
                                                         rocblas_int*      info,
                                                         const rocblas_int batch_count);

/*! \brief GETRF_STRIDED_BATCHED computes the LU factorization of a batch of general m-by-n matrices
    using partial pivoting with row interchanges.

    \details
    (This is the right-looking Level 3 BLAS version of the algorithm).

    The factorization of matrix A_i in the batch has the form

        A_i = P_i * L_i * U_i

    where P_i is a permutation matrix, L_i is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U_i is upper
    triangular (upper trapezoidal if m < n).

    @param[in]
    handle    rocblas_handle.
    @param[in]
    m         rocblas_int. m >= 0.\n
              The number of rows of all matrices A_i in the batch.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The number of colums of all matrices A_i in the batch.
    @param[inout]
    A         pointer to type. Array on the GPU (the size depends on the value of strideA).\n
              On entry, in contains the m-by-n matrices A_i to be factored.
              On exit, the factors L_i and U_i from the factorization.
              The unit diagonal elements of L_i are not stored.
    @param[in]
    lda       rocblas_int. lda >= m.\n
              Specifies the leading dimension of matrices A_i.
    @param[in]
    strideA   rocblas_int.\n
              Stride from the start of one matrix A_i and the next one A_(i+1).
              There is no restriction for the value of strideA. Normal use case is strideA >= lda*n
    @param[out]
    ipiv      pointer to rocblas_int. Array on the GPU (the size depends on the value of strideP).\n
              Contains the vectors of pivots indices ipiv_i (corresponding to A_i).
              Dimension of ipiv_i is min(m,n).
              Elements of ipiv_i are 1-based indices.
              For each instance A_i in the batch and for 1 <= j <= min(m,n), the row j of the
              matrix A_i was interchanged with row ipiv_i(j).
              Matrix P_i of the factorization can be derived from ipiv_i.
    @param[in]
    strideP   rocblas_int.\n
              Stride from the start of one vector ipiv_i to the next one ipiv_(i+1).
              There is no restriction for the value of strideP. Normal use case is strideP >= min(m,n).
    @param[out]
    info      pointer to rocblas_int. Array of batch_count integers on the GPU.\n
              If info_i = 0, succesful exit for factorization of A_i.
              If info_i = j > 0, U_i is singular. U_i(j,j) is the first zero pivot.
    @param[in]
    batch_count rocblas_int. batch_count >= 0.\n
                Number of matrices in the batch.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgetrf_strided_batched(rocblas_handle    handle,
                                                                 const rocblas_int m,
                                                                 const rocblas_int n,
                                                                 float*            A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_int strideA,
                                                                 rocblas_int*      ipiv,
                                                                 const rocblas_int strideP,
                                                                 rocblas_int*      info,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgetrf_strided_batched(rocblas_handle    handle,
                                                                 const rocblas_int m,
                                                                 const rocblas_int n,
                                                                 double*           A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_int strideA,
                                                                 rocblas_int*      ipiv,
                                                                 const rocblas_int strideP,
                                                                 rocblas_int*      info,
                                                                 const rocblas_int batch_count);

/*! \brief GEQR2 computes a QR factorization of a general m-by-n matrix A.

    \details
    (This is the unblocked version of the algorithm).

    The factorization has the form

        A =  Q * R

    where R is upper triangular (upper trapezoidal if m < n), and Q is
    an orthogonal matrix represented as the product of Householder matrices

        Q = H(1) * H(2) * ... * H(k), with k = min(m,n)

    Each Householder matrix H(i), for i = 1,2,...,k, is given by

        H(i) = I - ipiv[i-1] * v(i) * v(i)'

    where the first i-1 elements of the Householder vector v(i) are zero, and v(i)[i] = 1.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    m         rocblas_int. m >= 0.\n
              The number of rows of the matrix A.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The number of colums of the matrix A.
    @param[inout]
    A         pointer to type. Array on the GPU of dimension lda*n.\n
              On entry, the m-by-n matrix to be factored.
              On exit, the elements on and above the diagonal contain the
              factor R; the elements below the diagonal are the m - i elements
              of vector v(i) for i = 1,2,...,min(m,n).
    @param[in]
    lda       rocblas_int. lda >= m.\n
              Specifies the leading dimension of A.
    @param[out]
    ipiv      pointer to type. Array on the GPU of dimension min(m,n).\n
              The scalar factors of the Householder matrices H(i).

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgeqr2(rocblas_handle    handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 float*            A,
                                                 const rocblas_int lda,
                                                 float*            ipiv);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgeqr2(rocblas_handle    handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 double*           A,
                                                 const rocblas_int lda,
                                                 double*           ipiv);

/*! \brief GEQR2_BATCHED computes the QR factorization of a batch of general m-by-n matrices.

    \details
    (This is the unblocked version of the algorithm).

    The factorization of matrix A_j in the batch has the form

        A_j =  Q_j * R_j

    where R_j is upper triangular (upper trapezoidal if m < n), and Q_j is
    an orthogonal matrix represented as the product of Householder matrices

        Q_j = H_j(1) * H_j(2) * ... * H_j(k), with k = min(m,n)

    Each Householder matrices H_j(i), for j = 1,2,...,batch_count, and i = 1,2,...,k, is given by

        H_j(i) = I - ipiv_j[i-1] * v_j(i) * v_j(i)'

    where the first i-1 elements of vector Householder vector v_j(i) are zero, and v_j(i)[i] = 1.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    m         rocblas_int. m >= 0.\n
              The number of rows of all the matrices A_j in the batch.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The number of colums of all the matrices A_j in the batch.
    @param[inout]
    A         Array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
              On entry, the m-by-n matrices A_j to be factored.
              On exit, the elements on and above the diagonal contain the
              factor R_j. The elements below the diagonal are the m - i elements
              of vector v_j(i) for i=1,2,...,min(m,n).
    @param[in]
    lda       rocblas_int. lda >= m.\n
              Specifies the leading dimension of matrices A_j.
    @param[out]
    ipiv      pointer to type. Array on the GPU (the size depends on the value of strideP).\n
              Contains the vectors ipiv_j of scalar factors of the
              Householder matrices H_j(i).
    @param[in]
    strideP   rocblas_int.\n
              Stride from the start of one vector ipiv_j to the next one ipiv_(j+1).
              There is no restriction for the value
              of strideP. Normal use is strideP >= min(m,n).
    @param[in]
    batch_count  rocblas_int. batch_count >= 0.\n
                 Number of matrices in the batch.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgeqr2_batched(rocblas_handle    handle,
                                                         const rocblas_int m,
                                                         const rocblas_int n,
                                                         float* const      A[],
                                                         const rocblas_int lda,
                                                         float*            ipiv,
                                                         const rocblas_int strideP,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgeqr2_batched(rocblas_handle    handle,
                                                         const rocblas_int m,
                                                         const rocblas_int n,
                                                         double* const     A[],
                                                         const rocblas_int lda,
                                                         double*           ipiv,
                                                         const rocblas_int strideP,
                                                         const rocblas_int batch_count);

/*! \brief GEQR2_STRIDED_BATCHED computes the QR factorization of a batch of general m-by-n matrices.

    \details
    (This is the unblocked version of the algorithm).

    The factorization of matrix A_j in the batch has the form

        A_j =  Q_j * R_j

    where R_j is upper triangular (upper trapezoidal if m < n), and Q_j is
    an orthogonal matrix represented as the product of Householder matrices

        Q_j = H_j(1) * H_j(2) * ... * H_j(k), with k = min(m,n)

    Each Householder matrices H_j(i), for j = 1,2,...,batch_count, and i = 1,2,...,k, is given by

        H_j(i) = I - ipiv_j[i-1] * v_j(i) * v_j(i)'

    where the first i-1 elements of vector Householder vector v_j(i) are zero, and v_j(i)[i] = 1.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    m         rocblas_int. m >= 0.\n
              The number of rows of all the matrices A_j in the batch.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The number of colums of all the matrices A_j in the batch.
    @param[inout]
    A         pointer to type. Array on the GPU (the size depends on the value of strideA).\n
              On entry, the m-by-n matrices A_j to be factored.
              On exit, the elements on and above the diagonal contain the
              factor R_j. The elements below the diagonal are the m - i elements
              of vector v_j(i) for i = 1,2,...,min(m,n).
    @param[in]
    lda       rocblas_int. lda >= m.\n
              Specifies the leading dimension of matrices A_j.
    @param[in]
    strideA   rocblas_int.\n
              Stride from the start of one matrix A_j and the next one A_(j+1).
              There is no restriction for the value of strideA. Normal use case is strideA >= lda*n.
    @param[out]
    ipiv      pointer to type. Array on the GPU (the size depends on the value of strideP).\n
              Contains the vectors ipiv_j of scalar factors of the
              Householder matrices H_j(i).
    @param[in]
    strideP   rocblas_int.\n
              Stride from the start of one vector ipiv_j to the next one ipiv_(j+1).
              There is no restriction for the value
              of strideP. Normal use is strideP >= min(m,n).
    @param[in]
    batch_count  rocblas_int. batch_count >= 0.\n
                 Number of matrices in the batch.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgeqr2_strided_batched(rocblas_handle    handle,
                                                                 const rocblas_int m,
                                                                 const rocblas_int n,
                                                                 float*            A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_int strideA,
                                                                 float*            ipiv,
                                                                 const rocblas_int strideP,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgeqr2_strided_batched(rocblas_handle    handle,
                                                                 const rocblas_int m,
                                                                 const rocblas_int n,
                                                                 double*           A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_int strideA,
                                                                 double*           ipiv,
                                                                 const rocblas_int strideP,
                                                                 const rocblas_int batch_count);

/*! \brief GEQRF computes a QR factorization of a general m-by-n matrix A.

    \details
    (This is the blocked version of the algorithm).

    The factorization has the form

        A =  Q * R

    where R is upper triangular (upper trapezoidal if m < n), and Q is
    an orthogonal matrix represented as the product of Householder matrices

        Q = H(1) * H(2) * ... * H(k), with k = min(m,n)

    Each Householder matrix H(i), for i = 1,2,...,k, is given by

        H(i) = I - ipiv[i-1] * v(i) * v(i)'

    where the first i-1 elements of the Householder vector v(i) are zero, and v(i)[i] = 1.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    m         rocblas_int. m >= 0.\n
              The number of rows of the matrix A.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The number of colums of the matrix A.
    @param[inout]
    A         pointer to type. Array on the GPU of dimension lda*n.\n
              On entry, the m-by-n matrix to be factored.
              On exit, the elements on and above the diagonal contain the
              factor R; the elements below the diagonal are the m - i elements
              of vector v(i) for i = 1,2,...,min(m,n).
    @param[in]
    lda       rocblas_int. lda >= m.\n
              Specifies the leading dimension of A.
    @param[out]
    ipiv      pointer to type. Array on the GPU of dimension min(m,n).\n
              The scalar factors of the Householder matrices H(i).

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgeqrf(rocblas_handle    handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 float*            A,
                                                 const rocblas_int lda,
                                                 float*            ipiv);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgeqrf(rocblas_handle    handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 double*           A,
                                                 const rocblas_int lda,
                                                 double*           ipiv);

/*! \brief GEQRF_BATCHED computes the QR factorization of a batch of general m-by-n matrices.

    \details
    (This is the blocked version of the algorithm).

    The factorization of matrix A_j in the batch has the form

        A_j =  Q_j * R_j

    where R_j is upper triangular (upper trapezoidal if m < n), and Q_j is
    an orthogonal matrix represented as the product of Householder matrices

        Q_j = H_j(1) * H_j(2) * ... * H_j(k), with k = min(m,n)

    Each Householder matrices H_j(i), for j = 1,2,...,batch_count, and i = 1,2,...,k, is given by

        H_j(i) = I - ipiv_j[i-1] * v_j(i) * v_j(i)'

    where the first i-1 elements of vector Householder vector v_j(i) are zero, and v_j(i)[i] = 1.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    m         rocblas_int. m >= 0.\n
              The number of rows of all the matrices A_j in the batch.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The number of colums of all the matrices A_j in the batch.
    @param[inout]
    A         Array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
              On entry, the m-by-n matrices A_j to be factored.
              On exit, the elements on and above the diagonal contain the
              factor R_j. The elements below the diagonal are the m - i elements
              of vector v_j(i) for i=1,2,...,min(m,n).
    @param[in]
    lda       rocblas_int. lda >= m.\n
              Specifies the leading dimension of matrices A_j.
    @param[out]
    ipiv      pointer to type. Array on the GPU (the size depends on the value of strideP).\n
              Contains the vectors ipiv_j of scalar factors of the
              Householder matrices H_j(i).
    @param[in]
    strideP   rocblas_int.\n
              Stride from the start of one vector ipiv_j to the next one ipiv_(j+1).
              There is no restriction for the value
              of strideP. Normal use is strideP >= min(m,n).
    @param[in]
    batch_count  rocblas_int. batch_count >= 0.\n
                 Number of matrices in the batch.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgeqrf_batched(rocblas_handle    handle,
                                                         const rocblas_int m,
                                                         const rocblas_int n,
                                                         float* const      A[],
                                                         const rocblas_int lda,
                                                         float*            ipiv,
                                                         const rocblas_int strideP,
                                                         const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgeqrf_batched(rocblas_handle    handle,
                                                         const rocblas_int m,
                                                         const rocblas_int n,
                                                         double* const     A[],
                                                         const rocblas_int lda,
                                                         double*           ipiv,
                                                         const rocblas_int strideP,
                                                         const rocblas_int batch_count);

/*! \brief GEQRF_STRIDED_BATCHED computes the QR factorization of a batch of general m-by-n matrices.

    \details
    (This is the blocked version of the algorithm).

    The factorization of matrix A_j in the batch has the form

        A_j =  Q_j * R_j

    where R_j is upper triangular (upper trapezoidal if m < n), and Q_j is
    an orthogonal matrix represented as the product of Householder matrices

        Q_j = H_j(1) * H_j(2) * ... * H_j(k), with k = min(m,n)

    Each Householder matrices H_j(i), for j = 1,2,...,batch_count, and i = 1,2,...,k, is given by

        H_j(i) = I - ipiv_j[i-1] * v_j(i) * v_j(i)'

    where the first i-1 elements of vector Householder vector v_j(i) are zero, and v_j(i)[i] = 1.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    m         rocblas_int. m >= 0.\n
              The number of rows of all the matrices A_j in the batch.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The number of colums of all the matrices A_j in the batch.
    @param[inout]
    A         pointer to type. Array on the GPU (the size depends on the value of strideA).\n
              On entry, the m-by-n matrices A_j to be factored.
              On exit, the elements on and above the diagonal contain the
              factor R_j. The elements below the diagonal are the m - i elements
              of vector v_j(i) for i = 1,2,...,min(m,n).
    @param[in]
    lda       rocblas_int. lda >= m.\n
              Specifies the leading dimension of matrices A_j.
    @param[in]
    strideA   rocblas_int.\n
              Stride from the start of one matrix A_j and the next one A_(j+1).
              There is no restriction for the value of strideA. Normal use case is strideA >= lda*n.
    @param[out]
    ipiv      pointer to type. Array on the GPU (the size depends on the value of strideP).\n
              Contains the vectors ipiv_j of scalar factors of the
              Householder matrices H_j(i).
    @param[in]
    strideP   rocblas_int.\n
              Stride from the start of one vector ipiv_j to the next one ipiv_(j+1).
              There is no restriction for the value
              of strideP. Normal use is strideP >= min(m,n).
    @param[in]
    batch_count  rocblas_int. batch_count >= 0.\n
                 Number of matrices in the batch.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgeqrf_strided_batched(rocblas_handle    handle,
                                                                 const rocblas_int m,
                                                                 const rocblas_int n,
                                                                 float*            A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_int strideA,
                                                                 float*            ipiv,
                                                                 const rocblas_int strideP,
                                                                 const rocblas_int batch_count);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgeqrf_strided_batched(rocblas_handle    handle,
                                                                 const rocblas_int m,
                                                                 const rocblas_int n,
                                                                 double*           A,
                                                                 const rocblas_int lda,
                                                                 const rocblas_int strideA,
                                                                 double*           ipiv,
                                                                 const rocblas_int strideP,
                                                                 const rocblas_int batch_count);

/*! \brief GETRS solves a system of n linear equations on n variables using the LU factorization computed by GETRF.

    \details
    It solves one of the following systems:

        A  * X = B (no transpose),
        A' * X = B (transpose),  or
        A* * X = B (conjugate transpose)

    depending on the value of trans.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    trans       rocblas_operation.\n
                Specifies the form of the system of equations.
    @param[in]
    n           rocblas_int. n >= 0.\n
                The order of the system, i.e. the number of columns of A.
    @param[in]
    nrhs        rocblas_int. nrhs >= 0.\n
                The number of right hand sides, i.e., the number of columns
                of the matrix B.
    @param[in]
    A           pointer to type. Array on the GPU of dimension lda*n.\n
                The factors L and U of the factorization A = P*L*U returned by GETRF.
    @param[in]
    lda         rocblas_int. lda >= n.\n
                The leading dimension of A.
    @param[in]
    ipiv        pointer to rocblas_int. Array on the GPU of dimension n.\n
                The pivot indices returned by GETRF.
    @param[in,out]
    B           pointer to type. Array on the GPU of dimension ldb*nrhs.\n
                On entry, the right hand side matrix B.
                On exit, the solution matrix X.
    @param[in]
    ldb         rocblas_int. ldb >= n.\n
                The leading dimension of B.

   ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_sgetrs(rocblas_handle     handle,
                                                 rocblas_operation  trans,
                                                 rocblas_int        n,
                                                 rocblas_int        nrhs,
                                                 const float*       A,
                                                 rocblas_int        lda,
                                                 const rocblas_int* ipiv,
                                                 float*             B,
                                                 rocblas_int        ldb);

ROCSOLVER_EXPORT rocblas_status rocsolver_dgetrs(rocblas_handle     handle,
                                                 rocblas_operation  trans,
                                                 rocblas_int        n,
                                                 rocblas_int        nrhs,
                                                 const double*      A,
                                                 rocblas_int        lda,
                                                 const rocblas_int* ipiv,
                                                 double*            B,
                                                 rocblas_int        ldb);

/*! \brief POTF2 computes the Cholesky factorization of a real symmetric
    positive definite matrix A.

    \details
    (This is the unblocked version of the algorithm).

    The factorization has the form:

        A = U' * U, or
        A = L  * L'

    depending on the value of uplo. U is an upper triangular matrix and L is lower triangular.

    @param[in]
    handle    rocblas_handle.
    @param[in]
    uplo      rocblas_fill.\n
              specifies whether the factorization is upper or lower triangular.
    @param[in]
    n         rocblas_int. n >= 0.\n
              The matrix dimensions.
    @param[inout]
    A         pointer to type. Array on the GPU of dimension lda*n.\n
              The matrix A to be factored.
    @param[in]
    lda       rocblas_int. lda >= n.\n
              specifies the leading dimension of A.

    ********************************************************************/

ROCSOLVER_EXPORT rocblas_status rocsolver_spotf2(
    rocblas_handle handle, rocblas_fill uplo, rocblas_int n, float* A, rocblas_int lda);

ROCSOLVER_EXPORT rocblas_status rocsolver_dpotf2(
    rocblas_handle handle, rocblas_fill uplo, rocblas_int n, double* A, rocblas_int lda);

#ifdef __cplusplus
}
#endif

#endif /* _ROCLAPACK_FUNCTIONS_H */
