#include "rocauxiliary_larft.hpp"

template <typename T>
rocblas_status rocsolver_larft_impl(rocblas_handle       handle,
                                    const rocblas_direct direct,
                                    const rocblas_int    n,
                                    const rocblas_int    k,
                                    T*                   V,
                                    const rocblas_int    ldv,
                                    T*                   tau,
                                    T*                   F,
                                    const rocblas_int    ldf)
{
    if(!handle)
        return rocblas_status_invalid_handle;

    //logging is missing ???

    if(n < 0 || k < 1 || ldv < n || ldf < k)
        return rocblas_status_invalid_size;
    if(!V || !tau || !F)
        return rocblas_status_invalid_pointer;

    rocblas_int stridev     = 0;
    rocblas_int stridet     = 0;
    rocblas_int stridef     = 0;
    rocblas_int batch_count = 1;

    return rocsolver_larft_template<T>(handle,
                                       direct,
                                       n,
                                       k,
                                       V,
                                       0, //shifted 0 entries
                                       ldv,
                                       stridev,
                                       tau,
                                       stridet,
                                       F,
                                       ldf,
                                       stridef,
                                       batch_count);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

ROCSOLVER_EXPORT rocblas_status rocsolver_slarft(rocblas_handle       handle,
                                                 const rocblas_direct direct,
                                                 const rocblas_int    n,
                                                 const rocblas_int    k,
                                                 float*               V,
                                                 const rocblas_int    ldv,
                                                 float*               tau,
                                                 float*               T,
                                                 const rocblas_int    ldt)
{
    return rocsolver_larft_impl<float>(handle, direct, n, k, V, ldv, tau, T, ldt);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_dlarft(rocblas_handle       handle,
                                                 const rocblas_direct direct,
                                                 const rocblas_int    n,
                                                 const rocblas_int    k,
                                                 double*              V,
                                                 const rocblas_int    ldv,
                                                 double*              tau,
                                                 double*              T,
                                                 const rocblas_int    ldt)
{
    return rocsolver_larft_impl<double>(handle, direct, n, k, V, ldv, tau, T, ldt);
}

} //extern C
