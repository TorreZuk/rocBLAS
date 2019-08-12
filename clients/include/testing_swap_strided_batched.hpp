/* ************************************************************************
 * Copyright 2018-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */

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
void testing_swap_strided_batched_bad_arg(const Arguments& arg)
{
    rocblas_int N           = 100;
    rocblas_int incx        = 1;
    rocblas_int incy        = 1;
    rocblas_int stridex     = 1;
    rocblas_int stridey     = 1;
    rocblas_int batch_count = 5;

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

    EXPECT_ROCBLAS_STATUS(rocblas_swap_strided_batched<T>(
                              handle, N, nullptr, incx, stridex, dy, incy, stridey, batch_count),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocblas_swap_strided_batched<T>(
                              handle, N, dx, incx, stridex, nullptr, incy, stridey, batch_count),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocblas_swap_strided_batched<T>(
                              nullptr, N, dx, incx, stridex, dy, incy, stridey, batch_count),
                          rocblas_status_invalid_handle);
}

template <typename T>
void testing_swap_strided_batched(const Arguments& arg)
{
    rocblas_int N           = arg.N;
    rocblas_int incx        = arg.incx;
    rocblas_int incy        = arg.incy;
    rocblas_int stridex     = arg.stride_x;
    rocblas_int stridey     = arg.stride_y;
    rocblas_int batch_count = arg.batch_count;

    rocblas_local_handle handle;

    size_t abs_incx = incx >= 0 ? incx : -incx;
    size_t abs_incy = incy >= 0 ? incy : -incy;

    // argument sanity check before allocating invalid memory
    if(stridex < N * abs_incx || stridey < N * abs_incy ||
       stridex < 0 || stridey < 0 || batch_count <= 0)
    {
        static const size_t safe_size = 100; //  arbitrarily set to 100
        device_vector<T>    dx(safe_size);
        device_vector<T>    dy(safe_size);
        if(!dx || !dy)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        EXPECT_ROCBLAS_STATUS(rocblas_swap_strided_batched<T>(
                                  handle, N, dx, incx, stridex, dy, incy, stridey, batch_count),
                              rocblas_status_invalid_size);
        return;
    }

    if(N <= 0)
    {
        static const size_t safe_size = 100; //  arbitrarily set to 100
        device_vector<T>    dx(safe_size);
        device_vector<T>    dy(safe_size);
        if(!dx || !dy)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        CHECK_ROCBLAS_ERROR(rocblas_swap_strided_batched<T>(
            handle, N, dx, incx, stridex, dy, incy, stridey, batch_count));
        return;
    }

    size_t size_x = (size_t)stridex;
    size_t size_y = (size_t)stridey;

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<T> hx(size_x * batch_count, 1);
    host_vector<T> hy(size_y * batch_count, 0);
    host_vector<T> hx_gold(size_x * batch_count, 0);
    host_vector<T> hy_gold(size_y * batch_count, 1); // swapped

    // Initial Data on CPU
    // rocblas_seedrand();
    // rocblas_init<T>(hx, 1, N, abs_incx);
    // make hy different to hx
    // for(size_t i = 0; i < N; i++)
    // {
    //     hy[i * abs_incy] = hx[i * abs_incx] + 1.0;
    // };

    // swap vector is easy in STL; hy_gold = hx: save a swap in hy_gold which will be output of CPU
    // BLAS
    // hx_gold = hx;
    // hy_gold = hy;

    // allocate memory on device
    device_vector<T> dx(size_x * batch_count);
    device_vector<T> dy(size_y * batch_count);
    if(!dx || !dy)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    size_t dataSizeX = sizeof(T) * size_x * batch_count;
    size_t dataSizeY = sizeof(T) * size_y * batch_count;

    // copy vector data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dx, hx, dataSizeX, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy, dataSizeY, hipMemcpyHostToDevice));

    double gpu_time_used, cpu_time_used;
    double rocblas_error = 0.0;

    if(arg.unit_check || arg.norm_check)
    {
        // GPU BLAS
        CHECK_ROCBLAS_ERROR(rocblas_swap_strided_batched<T>(
            handle, N, dx, incx, stridex, dy, incy, stridey, batch_count));
        CHECK_HIP_ERROR(hipMemcpy(hx, dx, dataSizeX, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hy, dy, dataSizeY, hipMemcpyDeviceToHost));

        // CPU BLAS
        cpu_time_used = get_time_us();
        //cblas_swap<T>(N, hx_gold, incx, hy_gold, incy);
        cpu_time_used = get_time_us() - cpu_time_used;

        if(arg.unit_check)
        {
            for(int i = 0; i < batch_count; i++)
            {
                unit_check_general<T>(1, N, abs_incx, hx_gold + i * stridex, hx + i * stridex);
                unit_check_general<T>(1, N, abs_incy, hy_gold + i * stridey, hy + i * stridey);
            }
        }

        if(arg.norm_check)
        {
            for(int i = 0; i < batch_count; i++)
            {
                rocblas_error = norm_check_general<T>(
                    'F', 1, N, abs_incx, hx_gold + i * stridex, hx + i * stridex);
                rocblas_error = norm_check_general<T>(
                    'F', 1, N, abs_incy, hy_gold + i * stridey, hy + i * stridey);
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
            rocblas_swap_strided_batched<T>(
                handle, N, dx, incx, stridex, dy, incy, stridey, batch_count);
        }

        gpu_time_used = get_time_us(); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_swap_strided_batched<T>(
                handle, N, dx, incx, stridex, dy, incy, stridey, batch_count);
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        std::cout << "N,incx,incy,stride_x,stride_y,batch_count,rocblas-us" << std::endl;
        std::cout << N << "," << incx << "," << incy << "," << stridex << "," << stridey << ","
                  << batch_count << "," << gpu_time_used << std::endl;
    }
}