FILES="grid/*"
EXE="/home/isaiah/scisim/build/rigidbody2dcli/rigidbody2d_cli"
for f in $FILES
do
    echo "Processing $f"
    $($EXE $f > "outs/"$f".out") 
done