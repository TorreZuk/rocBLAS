template <rocblas_int NB, typename T, typename U, typename V>
rocblas_status rocblas_scal_template(rocblas_handle handle,
                                     rocblas_int    n,
                                     const V*       alpha,
                                     rocblas_stride stride_alpha,
                                     U              x,
                                     rocblas_int    offsetx,
                                     rocblas_int    incx,
                                     rocblas_stride stridex,
                                     rocblas_int    batch_count);
