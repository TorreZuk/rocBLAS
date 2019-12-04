/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <boost/program_options.hpp>
#include <iostream>
#include <stdio.h>

#include "testing_getf2_getrf.hpp"
//#include "testing_getf2_getrf_batched.hpp"
//#include "testing_getf2_getrf_strided_batched.hpp"
//#include "testing_geqr2_geqrf.hpp"
//#include "testing_geqr2_geqrf_batched.hpp"
//#include "testing_geqr2_geqrf_strided_batched.hpp"
//#include "testing_getrs.hpp"
//#include "testing_potf2.hpp"
//#include "testing_larfg.hpp"
//#include "testing_larf.hpp"
//#include "testing_larft.hpp"
//#include "testing_larfb.hpp"
//#include "testing_laswp.hpp"
//#include "utility.h"

namespace po = boost::program_options;

int main(int argc, char* argv[])
{
    Arguments argus;

    //disable unit_check in client benchmark, it is only
    // used in gtest unit test
    argus.unit_check = 0;

    // enable timing check,otherwise no performance data collected
    argus.timing = 1;

    std::string function;
    char        precision;

    rocblas_int device_id;

    po::options_description desc("rocsolver client command line options");
    desc.add_options()("help,h", "produces this help message")

        ("sizem,m",
         po::value<rocblas_int>(&argus.M)->default_value(1024),
         "Specific matrix size testing: the number of rows of a matrix.")

            ("sizen,n",
             po::value<rocblas_int>(&argus.N)->default_value(1024),
             "Specific matrix/vector/order size testing: the number of columns of a matrix,"
             "or the order of a system or transformation.")

                ("sizek,k",
                 po::value<rocblas_int>(&argus.K)->default_value(1024),
                 "Specific...  the number of columns in "
                 "A & C  and rows in B.")

                    ("lda",
                     po::value<rocblas_int>(&argus.lda)->default_value(1024),
                     "Specific leading dimension of matrix A, is only applicable to "
                     "BLAS-2 & BLAS-3: the number of rows.")

                        ("ldb",
                         po::value<rocblas_int>(&argus.ldb)->default_value(1024),
                         "Specific leading dimension of matrix B, is only applicable to BLAS-2 & "
                         "BLAS-3: the number "
                         "of rows.")

                            ("ldc",
                             po::value<rocblas_int>(&argus.ldc)->default_value(1024),
                             "Specific leading dimension of matrix C, is only applicable to BLAS-2 "
                             "& "
                             "BLAS-3: the number of rows.")

                                ("ldv",
                                 po::value<rocblas_int>(&argus.ldv)->default_value(1024),
                                 "Specific leading dimension.")

                                    ("ldt",
                                     po::value<rocblas_int>(&argus.ldt)->default_value(1024),
                                     "Specific leading dimension.")

                                        ("stride_a",
                                         po::value<rocblas_int>(&argus.stride_a)
                                             ->default_value(1024 * 1024),
                                         "Specific stride of strided_batched matrix A, is only "
                                         "applicable to strided batched"
                                         "BLAS-2 and BLAS-3: second dimension * leading dimension.")

                                            ("stride_b",
                                             po::value<rocblas_int>(&argus.stride_b)
                                                 ->default_value(1024 * 1024),
                                             "Specific stride of strided_batched matrix B, is only "
                                             "applicable to strided batched"
                                             "BLAS-2 and BLAS-3: second dimension * leading "
                                             "dimension.")

                                                ("stride_c",
                                                 po::value<rocblas_int>(&argus.stride_c)
                                                     ->default_value(1024 * 1024),
                                                 "Specific stride of strided_batched matrix B, is "
                                                 "only applicable to strided batched"
                                                 "BLAS-2 and BLAS-3: second dimension * leading "
                                                 "dimension.")

                                                    ("stride_p",
                                                     po::value<rocblas_int>(&argus.stride_p)
                                                         ->default_value(1024),
                                                     "Specific stride of batched pivots vector "
                                                     "Ipiv, is only applicable to batched and "
                                                     "strided_batched"
                                                     "factorizations: min(first dimension, second "
                                                     "dimension).")

                                                        ("incx",
                                                         po::value<rocblas_int>(&argus.incx)
                                                             ->default_value(1),
                                                         "increment between values in x vector")

                                                            ("incy",
                                                             po::value<rocblas_int>(&argus.incy)
                                                                 ->default_value(1),
                                                             "increment between values in y vector")

                                                                ("alpha",
                                                                 po::value<double>(&argus.alpha)
                                                                     ->default_value(1.0),
                                                                 "specifies the scalar alpha")

                                                                    ("beta",
                                                                     po::value<double>(&argus.beta)
                                                                         ->default_value(0.0),
                                                                     "specifies the scalar beta")

                                                                        ("function,f",
                                                                         po::value<std::string>(
                                                                             &function)
                                                                             ->default_value(
                                                                                 "potf2"),
                                                                         "LAPACK function to test. "
                                                                         "Options: potf2, getf2, "
                                                                         "getrf, getrs")

                                                                            ("precision,r",
                                                                             po::value<char>(
                                                                                 &precision)
                                                                                 ->default_value(
                                                                                     's'),
                                                                             "Options: h,s,d,c,z")

                                                                                ("transposeA",
                                                                                 po::value<char>(
                                                                                     &argus.transA)
                                                                                     ->default_value(
                                                                                         'N'),
                                                                                 "N = no "
                                                                                 "transpose, T = "
                                                                                 "transpose, C = "
                                                                                 "conjugate "
                                                                                 "transpose")

                                                                                    ("transposeB",
                                                                                     po::value<
                                                                                         char>(
                                                                                         &argus
                                                                                              .transB)
                                                                                         ->default_value(
                                                                                             'N'),
                                                                                     "N = no "
                                                                                     "transpose, T "
                                                                                     "= transpose, "
                                                                                     "C = "
                                                                                     "conjugate "
                                                                                     "transpose")

                                                                                        ("transpose"
                                                                                         "H",
                                                                                         po::value<
                                                                                             char>(
                                                                                             &argus
                                                                                                  .transH)
                                                                                             ->default_value(
                                                                                                 'N'),
                                                                                         "N = no "
                                                                                         "transpose"
                                                                                         ", T = "
                                                                                         "transpose"
                                                                                         ", C = "
                                                                                         "conjugate"
                                                                                         " transpos"
                                                                                         "e")

                                                                                            ("side",
                                                                                             po::value<
                                                                                                 char>(
                                                                                                 &argus
                                                                                                      .side)
                                                                                                 ->default_value(
                                                                                                     'L'),
                                                                                             "L = "
                                                                                             "left,"
                                                                                             " R = "
                                                                                             "right"
                                                                                             ". "
                                                                                             "Only "
                                                                                             "appli"
                                                                                             "cable"
                                                                                             " to "
                                                                                             "certa"
                                                                                             "in "
                                                                                             "routi"
                                                                                             "nes")

                                                                                                ("u"
                                                                                                 "p"
                                                                                                 "l"
                                                                                                 "o",
                                                                                                 po::value<
                                                                                                     char>(
                                                                                                     &argus
                                                                                                          .uplo)
                                                                                                     ->default_value(
                                                                                                         'U'),
                                                                                                 "U"
                                                                                                 " "
                                                                                                 "="
                                                                                                 " "
                                                                                                 "u"
                                                                                                 "p"
                                                                                                 "p"
                                                                                                 "e"
                                                                                                 "r"
                                                                                                 ","
                                                                                                 " "
                                                                                                 "L"
                                                                                                 " "
                                                                                                 "="
                                                                                                 " "
                                                                                                 "l"
                                                                                                 "o"
                                                                                                 "w"
                                                                                                 "e"
                                                                                                 "r"
                                                                                                 "."
                                                                                                 " "
                                                                                                 "O"
                                                                                                 "n"
                                                                                                 "l"
                                                                                                 "y"
                                                                                                 " "
                                                                                                 "a"
                                                                                                 "p"
                                                                                                 "p"
                                                                                                 "l"
                                                                                                 "i"
                                                                                                 "c"
                                                                                                 "a"
                                                                                                 "b"
                                                                                                 "l"
                                                                                                 "e"
                                                                                                 " "
                                                                                                 "t"
                                                                                                 "o"
                                                                                                 " "
                                                                                                 "c"
                                                                                                 "e"
                                                                                                 "r"
                                                                                                 "t"
                                                                                                 "a"
                                                                                                 "i"
                                                                                                 "n"
                                                                                                 " "
                                                                                                 "r"
                                                                                                 "o"
                                                                                                 "u"
                                                                                                 "t"
                                                                                                 "i"
                                                                                                 "n"
                                                                                                 "e"
                                                                                                 "s") // xsymv xsyrk xsyr2k xtrsm
        // xtrmm
        ("diag",
         po::value<char>(&argus.diag)->default_value('N'),
         "U = unit diagonal, N = non unit diagonal. Only applicable to certain routines") // xtrsm
        // xtrmm
        ("direct",
         po::value<char>(&argus.direct)->default_value('F'),
         "F = forward, B = backward. Only applicable to certain routines") // xtrsm

        ("batch",
         po::value<rocblas_int>(&argus.batch_count)->default_value(1),
         "Number of matrices. Only applicable to batched routines") // xtrsm xtrmm xgemm

        ("verify,v",
         po::value<rocblas_int>(&argus.norm_check)->default_value(0),
         "Validate GPU results with CPU? 0 = No, 1 = Yes (default: No)")

            ("iters,i",
             po::value<rocblas_int>(&argus.iters)->default_value(10),
             "Iterations to run inside timing loop")

                ("device",
                 po::value<rocblas_int>(&device_id)->default_value(0),
                 "Set default device to be used for subsequent program runs");
    // clang-format on

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if(vm.count("help"))
    {
        std::cout << desc << std::endl;
        return 0;
    }

    if(precision != 'h' && precision != 's' && precision != 'd' && precision != 'c'
       && precision != 'z')
    {
        std::cerr << "Invalid value for --precision" << std::endl;
        return -1;
    }

    // Device Query
    rocblas_int device_count = query_device_property();

    if(device_count <= device_id)
    {
        printf("Error: invalid device ID. There may not be such device ID. Will "
               "exit \n");
        return -1;
    }
    else
    {
        set_device(device_id);
    }

    if(argus.M < 0 || argus.N < 0 || argus.K < 0)
    {
        printf("Invalide matrix dimension\n");
    }

    if(function == "potf2")
    {
        //    if (precision == 's')
        //      testing_potf2<float>(argus);
        //    else if (precision == 'd')
        //      testing_potf2<double>(argus);
    }
    /*  else if (function == "laswp") {
    if (precision == 's')
      testing_laswp<float>(argus);
    else if (precision == 'd')
      testing_laswp<double>(argus);
  }
  else if (function == "larft") {
    if (precision == 's')
      testing_larft<float>(argus);
    else if (precision == 'd')
      testing_larft<double>(argus);
  }
  else if (function == "larfg") {
    if (precision == 's')
      testing_larfg<float>(argus);
    else if (precision == 'd')
      testing_larfg<double>(argus);
  }
  else if (function == "larf") {
    if (precision == 's')
      testing_larf<float>(argus);
    else if (precision == 'd')
      testing_larf<double>(argus);
  }*/
    else if(function == "getf2")
    {
        if(precision == 's')
            testing_getf2_getrf<float, 0>(argus);
        else if(precision == 'd')
            testing_getf2_getrf<double, 0>(argus);
    }
    /*  else if (function == "getf2_batched") {
    if (precision == 's')
      testing_getf2_getrf_batched<float,0>(argus);
    else if (precision == 'd')
      testing_getf2_getrf_batched<double,0>(argus);
  }
  else if (function == "getf2_strided_batched") {
    if (precision == 's')
      testing_getf2_getrf_strided_batched<float,0>(argus);
    else if (precision == 'd')
      testing_getf2_getrf_strided_batched<double,0>(argus);
  }
  else if (function == "geqr2") {
    if (precision == 's')
      testing_geqr2_geqrf<float,0>(argus);
    else if (precision == 'd')
      testing_geqr2_geqrf<double,0>(argus);
  }
  else if (function == "geqr2_batched") {
    if (precision == 's')
      testing_geqr2_geqrf_batched<float,0>(argus);
    else if (precision == 'd')
      testing_geqr2_geqrf_batched<double,0>(argus);
  }
  else if (function == "geqr2_strided_batched") {
    if (precision == 's')
      testing_geqr2_geqrf_strided_batched<float,0>(argus);
    else if (precision == 'd')
      testing_geqr2_geqrf_strided_batched<double,0>(argus);
  }*/

#if BUILD_WITH_TENSILE
    else if(function == "getrf")
    {
        if(precision == 's')
            testing_getf2_getrf<float, 1>(argus);
        else if(precision == 'd')
            testing_getf2_getrf<double, 1>(argus);
    }
    else if(function == "getrf_batched")
    {
        if(precision == 's')
            testing_getf2_getrf_batched<float, 1>(argus);
        else if(precision == 'd')
            testing_getf2_getrf_batched<double, 1>(argus);
    }
    else if(function == "getrf_strided_batched")
    {
        if(precision == 's')
            testing_getf2_getrf_strided_batched<float, 1>(argus);
        else if(precision == 'd')
            testing_getf2_getrf_strided_batched<double, 1>(argus);
    }
    else if(function == "geqrf")
    {
        if(precision == 's')
            testing_geqr2_geqrf<float, 1>(argus);
        else if(precision == 'd')
            testing_geqr2_geqrf<double, 1>(argus);
    }
    else if(function == "geqrf_batched")
    {
        if(precision == 's')
            testing_geqr2_geqrf_batched<float, 1>(argus);
        else if(precision == 'd')
            testing_geqr2_geqrf_batched<double, 1>(argus);
    }
    else if(function == "geqrf_strided_batched")
    {
        if(precision == 's')
            testing_geqr2_geqrf_strided_batched<float, 1>(argus);
        else if(precision == 'd')
            testing_geqr2_geqrf_strided_batched<double, 1>(argus);
    }
    else if(function == "getrs")
    {
        if(precision == 's')
            testing_getrs<float>(argus);
        else if(precision == 'd')
            testing_getrs<double>(argus);
    }
    else if(function == "larfb")
    {
        if(precision == 's')
            testing_larfb<float>(argus);
        else if(precision == 'd')
            testing_larfb<double>(argus);
    }
#endif
    else
    {
        printf("Invalid value for --function \n");
        return -1;
    }

    return 0;
}
