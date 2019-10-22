# multi-line

BEGIN { RS = "\n\n+"; FPAT = "^template.*_template(.*).*{.*" }

{
    if (NF == 1){
        gsub(/{.*/,"",$1); # strip out body
        gsub(/)/,");",$1)
        print $1;   
    }
}