/* ************************************************************************
 * Copyright 2018-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#pragma once

#include "cblas_interface.hpp"
#include "norm.hpp"
#include "rocblas.hpp"
#include "rocblas_init.hpp"
#include "rocblas_math.hpp"
#include "rocblas_random.hpp"
#include "rocblas_test.hpp"
#include "rocblas_vector.hpp"
#include "unit.hpp"
#include "utility.hpp"

template <typename T>
void testing_swap_batched_bad_arg(const Arguments& arg)
{
    rocblas_int         N         = 100;
    rocblas_int         incx      = 1;
    rocblas_int         incy      = 1;
    rocblas_int         batch_count = 1;

    rocblas_local_handle handle;

    T**                 dxt;
    hipMalloc(&dxt, sizeof(T*));
    T**                 dyt;
    hipMalloc(&dyt, sizeof(T*));
    if(!dxt || !dyt)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    EXPECT_ROCBLAS_STATUS(rocblas_swap_batched<T>(handle, N, nullptr, incx, dyt, incy, batch_count),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocblas_swap_batched<T>(handle, N, dxt, incx, nullptr, incy, batch_count),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocblas_swap_batched<T>(nullptr, N, dxt, incx, dyt, incy, batch_count),
                          rocblas_status_invalid_handle);

    hipFree(dxt);
    hipFree(dyt);
}

template <typename T>
void testing_swap_batched(const Arguments& arg)
{
    rocblas_int          N    = arg.N;
    rocblas_int          incx = arg.incx;
    rocblas_int          incy = arg.incy;
    rocblas_int          batch_count = arg.batch_count;

    rocblas_local_handle handle;

    // argument sanity check before allocating invalid memory
    if(batch_count <= 0)
    {
        static const size_t safe_size = 100; //  arbitrarily set to 100

        T**                 dxt;
        hipMalloc(&dxt, sizeof(T*));
        T**                 dyt;
        hipMalloc(&dyt, sizeof(T*));
        if(!dxt || !dyt)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        EXPECT_ROCBLAS_STATUS(rocblas_swap_batched<T>(handle, N, dxt, incx, dyt, incy, batch_count),
                              rocblas_status_invalid_size);
        CHECK_HIP_ERROR(hipFree(dxt));
        CHECK_HIP_ERROR(hipFree(dyt));
        return;
    }

    if(N <= 0)
    {
        static const size_t safe_size = 100; //  arbitrarily set to 100
        
        T**                 dxt;
        hipMalloc(&dxt, sizeof(T*));
        T**                 dyt;
        hipMalloc(&dyt, sizeof(T*));
        if(!dxt || !dyt)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR(rocblas_swap_batched<T>(handle, N, dxt, incx, dyt, incy, batch_count));
        CHECK_HIP_ERROR(hipFree(dxt));
        CHECK_HIP_ERROR(hipFree(dyt));
        return;
    }

    ssize_t abs_incx = (incx >= 0) ? incx : -incx;
    ssize_t abs_incy = (incy >= 0) ? incy : -incy;

    size_t size_x   = N * abs_incx;
    size_t size_y   = N * abs_incy;

    // Naming: dx_pvec is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<T> hx[batch_count]; // (batch_count*size_x, 1);
    host_vector<T> hy[batch_count]; //(batch_count*size_y, 0);
    host_vector<T> hx_gold[batch_count];//(batch_count*size_x, 0);
    host_vector<T> hy_gold[batch_count];//(batch_count*size_y, 1);// swapped

    for(int i = 0; i < batch_count; i++)
    {
        hx[i]    = host_vector<T>(size_x, 0);
        hy[i]    = host_vector<T>(size_y, 1);
        hx_gold[i]    = host_vector<T>(size_x, 1);
        hy_gold[i]    = host_vector<T>(size_y, 0); // gold swapped
    }

    // Initial Data on CPU
    //rocblas_seedrand();
    //rocblas_init<T>(hx, 1, N, abs_incx);
    // make hy different to hx
    // for(size_t i = 0; i < N; i++)
    // {
    //     hy[i * abs_incy] = hx[i * abs_incx] + 1.0;
    // };

    // swap vector is easy in STL; hy_gold = hx: save a swap in hy_gold which will be output of CPU
    // BLAS
    //hx_gold = hx;
    //hy_gold = hy;


    T** hdx = new T*[batch_count]; // must create device ptr array on host
    T** hdy = new T*[batch_count]; // gpu pointers on cpu
    for(int i = 0; i < batch_count; i++)
    {
        hipMalloc(&hdx[i], size_x*sizeof(T));
        hipMalloc(&hdy[i], size_y*sizeof(T)); 
    }

    // copy data from host to device
    for(int i = 0; i < batch_count; i++)
    {
        CHECK_HIP_ERROR(hipMemcpy(hdx[i], hx[i], sizeof(T) * size_x, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(hdy[i], hy[i], sizeof(T) * size_y, hipMemcpyHostToDevice));
    }

    // vector pointers on gpu
    T** dx_pvec;
    T** dy_pvec;    
    CHECK_HIP_ERROR(hipMalloc(&dx_pvec, batch_count * sizeof(T*)));
    CHECK_HIP_ERROR(hipMalloc(&dy_pvec, batch_count * sizeof(T*)));
    if(!dx_pvec || !dy_pvec)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // copy gpu vector pointers from host to device pointer array
    CHECK_HIP_ERROR(hipMemcpy(dx_pvec, &hdx[0], sizeof(T*) * batch_count, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy_pvec, &hdy[0], sizeof(T*) * batch_count, hipMemcpyHostToDevice));


    double gpu_time_used, cpu_time_used;
    double rocblas_error = 0.0;

    if(arg.unit_check || arg.norm_check)
    {
        // GPU BLAS
        // CHECK_HIP_ERROR(hipMemcpy(dx_pvec, hx, sizeof(T) * size_x, hipMemcpyHostToDevice));
        // CHECK_HIP_ERROR(hipMemcpy(dy_pvec, hy, sizeof(T) * size_y, hipMemcpyHostToDevice));
        CHECK_ROCBLAS_ERROR(rocblas_swap_batched<T>(handle, N, dx_pvec, incx, dy_pvec, incy, batch_count));

        // copy data from device to CPU 
        for(int i = 0; i < batch_count; i++)
        {
            CHECK_HIP_ERROR(hipMemcpy(hx[i], hdx[i], sizeof(T) * size_x, hipMemcpyDeviceToHost));
            CHECK_HIP_ERROR(hipMemcpy(hy[i], hdy[i], sizeof(T) * size_y, hipMemcpyDeviceToHost));
        }

        // CPU BLAS
        cpu_time_used = get_time_us();
        //cblas_swap_batched<T>(N, hx_gold, incx, hy_gold, incy);
        cpu_time_used = get_time_us() - cpu_time_used;

        if(arg.unit_check)
        {
            for(int i = 0; i < batch_count; i++)
            {
                unit_check_general<T>(1, N, abs_incx, hx_gold[i], hx[i]);
                unit_check_general<T>(1, N, abs_incy, hy_gold[i], hy[i]);
            }
        }

        if(arg.norm_check)
        {
            for(int i = 0; i < batch_count; i++)
            {
                rocblas_error = norm_check_general<T>('F', 1, N, abs_incx, hx_gold[i], hx[i]);
                rocblas_error = norm_check_general<T>('F', 1, N, abs_incy, hy_gold[i], hy[i]);
            }
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = 100; 
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_swap_batched<T>(handle, N, dx_pvec, incx, dy_pvec, incy, batch_count);
        }

        gpu_time_used = get_time_us(); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_swap_batched<T>(handle, N, dx_pvec, incx, dy_pvec, incy, batch_count);
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        std::cout << "N,incx,incy,batch_count,rocblas-us" << std::endl;
        std::cout << N << "," << incx << "," << incy << "," << batch_count << "," << gpu_time_used << std::endl;
    }

    for(int i = 0; i < batch_count; i++)
    {
        CHECK_HIP_ERROR(hipFree(hdx[i]));
        CHECK_HIP_ERROR(hipFree(hdy[i])); // gpu pointers on cpu
    }
    delete [] hdx;
    delete [] hdy;
    CHECK_HIP_ERROR(hipFree(dx_pvec));
    CHECK_HIP_ERROR(hipFree(dy_pvec));
}
