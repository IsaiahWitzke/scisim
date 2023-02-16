import os
import sys

PROJ_BASE = '/home/isaiah/scisim/'

def run_rigidbody2dcli_dir(
    in_dir,
    out_dir,
    exe = PROJ_BASE + 'build/rigidbody2dcli/rigidbody2d_cli'
):
    for f in os.listdir(in_dir):
        os.system(f"{exe} -e 0.05 {in_dir}{f} > {out_dir}{f}.out")

def run_rigidbody2dcli(
    in_file,
    out_file,
    exe = PROJ_BASE + 'build/rigidbody2dcli/rigidbody2d_cli'
):
    os.system(f"{exe} -e 0.05 {in_file} > {out_file}")