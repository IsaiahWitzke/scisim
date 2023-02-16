import os
import sys

if len(sys.argv) == 4:
    INPUT_DIR = sys.argv[2]
    OUT_DIR = sys.argv[3]
else:
    print("Usage: python generate_balls.py <path to output folder>")
    exit()

if 'RIGID_BODY_2D_CLI' in os.environ:
    RIGID_BODY_2D_CLI_PATH = os.environ['RIGID_BODY_2D_CLI']
else:
    RIGID_BODY_2D_CLI_PATH = 


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
