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

    static const size_t safe_size = 100; //  arbitrarily set to 100

    rocblas_local_handle handle;

    // allocate memory on device
    device_vector<T> dx(safe_size);
    device_vector<T> dy(safe_size);
    if(!dx || !dy)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    EXPECT_ROCBLAS_STATUS(rocblas_swap_batched<T>(handle, N, nullptr, incx, &dy, incy, batch_count),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocblas_swap_batched<T>(handle, N, &dx, incx, nullptr, incy, batch_count),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocblas_swap_batched<T>(nullptr, N, &dx, incx, &dy, incy, batch_count),
                          rocblas_status_invalid_handle);
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
    if(N <= 0 || batch_count <= 0)
    {
        static const size_t safe_size = 100; //  arbitrarily set to 100
        device_vector<T>    dx(safe_size);
        device_vector<T>    dy(safe_size);
        std::vector< T* > dxvec;
        dxvec.push_back(&dx[0]);
        std::vector< T* > dyvec;
        dyvec.push_back(&dy[0]);

        if(!dx || !dy)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        CHECK_ROCBLAS_ERROR(rocblas_swap_batched<T>(handle, N, &dxvec[0], incx, &dyvec[0], incy, batch_count));
        return;
    }

    size_t abs_incx = incx >= 0 ? incx : -incx;
    size_t abs_incy = incy >= 0 ? incy : -incy;
    size_t size_x   = N * abs_incx;
    size_t size_y   = N * abs_incy;

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<T> hx[batch_count];
    host_vector<T> hy[batch_count];
    host_vector<T> hx_gold[batch_count];
    host_vector<T> hy_gold[batch_count];

    for(int i = 0; i < batch_count; i++)
    {
        hx[i]    = host_vector<T>(size_x, i);
        hy[i]    = host_vector<T>(size_y, 0);
        hx_gold[i] = host_vector<T>(size_x, 0); // swapped
        hy_gold[i] = host_vector<T>(size_y, i);
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

    // allocate memory on device
    device_vector<T> dxBlock(batch_count*size_x);
    device_vector<T> dyBlock(batch_count*size_y);
    if(!dxBlock || !dyBlock)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    device_vector<T*> dx(batch_count);
    device_vector<T*> dy(batch_count);

    // copy data from CPU to device
    for(int i = 0; i < batch_count; i++)
    {
        dx[i]    = dxBlock + i*size_x;
        dy[i]    = dyBlock + i*size_y;
        CHECK_HIP_ERROR(hipMemcpy(dx[i], hx[i], sizeof(T) * size_x, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dy[i], hy[i], sizeof(T) * size_y, hipMemcpyHostToDevice));
    }

    double gpu_time_used, cpu_time_used;
    double rocblas_error = 0.0;

    if(arg.unit_check || arg.norm_check)
    {
        // GPU BLAS
        // CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(T) * size_x, hipMemcpyHostToDevice));
        // CHECK_HIP_ERROR(hipMemcpy(dy, hy, sizeof(T) * size_y, hipMemcpyHostToDevice));
        CHECK_ROCBLAS_ERROR(rocblas_swap_batched<T>(handle, N, dx, incx, dy, incy, batch_count));

        // copy data from device to CPU 
        for(int i = 0; i < batch_count; i++)
        {
            CHECK_HIP_ERROR(hipMemcpy(hx[i], dx[i], sizeof(T) * size_x, hipMemcpyDeviceToHost));
            CHECK_HIP_ERROR(hipMemcpy(hy[i], dy[i], sizeof(T) * size_y, hipMemcpyDeviceToHost));
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
            rocblas_swap_batched<T>(handle, N, dx, incx, dy, incy, batch_count);
        }

        gpu_time_used = get_time_us(); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_swap_batched<T>(handle, N, dx, incx, dy, incy, batch_count);
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        std::cout << "N,incx,incy,batch_count,rocblas-us" << std::endl;
        std::cout << N << "," << incx << "," << incy << "," << batch_count << "," << gpu_time_used << std::endl;
    }
}
