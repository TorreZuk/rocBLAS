# multi-line
BEGIN { RS = "\n\n+"; FPAT = "^template ROCBLAS_EXPORT(.*);" }

{
    if (NF == 1) {
      gsub("template ROCBLAS_EXPORT","extern template",$1)
      print $1;
    }
}