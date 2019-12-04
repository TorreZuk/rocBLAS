/* ************************************************************************
 * Copyright 2018-2019 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "rocblas_data.hpp"
#include "rocblas_datatype2string.hpp"
#include "rocblas_test.hpp"
#include "testing_getf2_getrf.hpp"
//#include "testing_getf2_getrf_batched.hpp"
//#include "testing_getf2_getrf_strided_batched.hpp"
#include "type_dispatch.hpp"
#include <cstring>
#include <type_traits>

namespace
{
    // possible test cases
    enum lu_test_type
    {
        GETRF,
        GETRF_BATCHED,
        GETRF_STRIDED_BATCHED,
        GETF2,
        GETF2_BATCHED,
        GETF2_STRIDED_BATCHED,
    };

    //test template
    template <template <typename...> class FILTER, lu_test_type LU_TYPE>
    struct lu_template : RocBLAS_Test<lu_template<FILTER, LU_TYPE>, FILTER>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return rocblas_simple_dispatch<lu_template::template type_filter_functor>(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            switch(LU_TYPE)
            {
            case GETRF:
                return !strcmp(arg.function, "getrf");
            case GETRF_BATCHED:
                return !strcmp(arg.function, "getrf_batched");
            case GETRF_STRIDED_BATCHED:
                return !strcmp(arg.function, "getrf_strided_batched");
            case GETF2:
                return !strcmp(arg.function, "getf2");
            case GETF2_BATCHED:
                return !strcmp(arg.function, "getf2_batched");
            case GETF2_STRIDED_BATCHED:
                return !strcmp(arg.function, "getf2_strided_batched");
            }
            return false;
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            RocBLAS_TestName<lu_template> name;

            name << rocblas_datatype2string(arg.a_type);

            /*            if(strstr(arg.function, "_bad_arg") != nullptr)
            {
                name << "_bad_arg";
            }
            else
            {
                name << '_' << arg.M << '_' << arg.N << '_' << arg.alpha << '_' << arg.incx;

                if(GER_TYPE == GER_STRIDED_BATCHED)
                    name << '_' << arg.stride_x;

                name << '_' << arg.incy;

                if(GER_TYPE == GER_STRIDED_BATCHED)
                    name << '_' << arg.stride_y;

                name << '_' << arg.lda;

                if(GER_TYPE == GER_STRIDED_BATCHED)
                    name << '_' << arg.stride_a;

                if(GER_TYPE == GER_STRIDED_BATCHED || GER_TYPE == GER_BATCHED)
                    name << '_' << arg.batch_count;
            }
*/
            return std::move(name);
        }
    };

    // By default, this test does not apply to any types.
    // The unnamed second parameter is used for enable_if below.
    template <typename, typename = void>
    struct lu_testing : rocblas_test_invalid
    {
    };

    // When the condition in the second argument is satisfied, the type combination
    // is valid. When the condition is false, this specialization does not apply.
    template <typename T>
    struct lu_testing<
        T,
        typename std::enable_if<std::is_same<T, float>{} || std::is_same<T, double>{}>::type>
        : rocblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "getf2"))
                testing_getf2_getrf<T, 0>(arg);
                //            else if(!strcmp(arg.function, "getf2_batched"))
                //                testing_getf2_getrf_batched<T,0>(arg);
                //            else if(!strcmp(arg.function, "getf2_strided_batched"))
                //                testing_getf2_getrf_strided_batched<T,0>(arg);
#if BUILD_WITH_TENSILE
            else if(!strcmp(arg.function, "getrf"))
                testing_getf2_getrf<T, 1>(arg);
            else if(!strcmp(arg.function, "getrf_batched"))
                testing_getf2_getrf_batched<T, 1>(arg);
            else if(!strcmp(arg.function, "getrf_strided_batched"))
                testing_getf2_getrf_strided_batched<T, 1>(arg);
#endif
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

#if BUILD_WITH_TENSILE
    using getrf = lu_template<lu_testing, GETRF>;
    TEST_P(getrf, lapack)
    {
        rocblas_simple_dispatch<lu_testing>(GetParam());
    }
    INSTANTIATE_TEST_CATEGORIES(getrf);

    using getrf_batched = lu_template<lu_testing, GETRF_BATCHED>;
    TEST_P(getrf_batched, lapack)
    {
        rocblas_simple_dispatch<lu_testing>(GetParam());
    }
    INSTANTIATE_TEST_CATEGORIES(getrf_batched);

    using getrf_strided_batched = lu_template<lu_testing, GETRF_STRIDED_BATCHED>;
    TEST_P(getrf_strided_batched, lapack)
    {
        rocblas_simple_dispatch<lu_testing>(GetParam());
    }
    INSTANTIATE_TEST_CATEGORIES(getrf_strided_batched);
#endif

    using getf2 = lu_template<lu_testing, GETF2>;
    TEST_P(getf2, lapack)
    {
        rocblas_simple_dispatch<lu_testing>(GetParam());
    }
    INSTANTIATE_TEST_CATEGORIES(getf2);

    using getf2_batched = lu_template<lu_testing, GETF2_BATCHED>;
    TEST_P(getf2_batched, lapack)
    {
        rocblas_simple_dispatch<lu_testing>(GetParam());
    }
    INSTANTIATE_TEST_CATEGORIES(getf2_batched);

    using getf2_strided_batched = lu_template<lu_testing, GETF2_STRIDED_BATCHED>;
    TEST_P(getf2_strided_batched, lapack)
    {
        rocblas_simple_dispatch<lu_testing>(GetParam());
    }
    INSTANTIATE_TEST_CATEGORIES(getf2_strided_batched);

} // namespace
