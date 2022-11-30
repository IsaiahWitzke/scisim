FILES="grid_v2/*"
EXE="/home/isaiah/scisim/build/rigidbody2dcli/rigidbody2d_cli"
for f in $FILES
do
    echo "Processing $f"
    $($EXE -e 0.01 $f > "outs/"$f".out") 
done