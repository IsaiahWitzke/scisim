import os
import sys

if len(sys.argv) == 3:
    INPUT_DIR = sys.argv[1]
    OUT_DIR = sys.argv[2]
else:
    print("Usage: python runner.py <path to input folder> <path to output folder>")
    exit()

if 'RIGID_BODY_2D_CLI' in os.environ:
    RIGID_BODY_2D_CLI_PATH = os.environ['RIGID_BODY_2D_CLI']
else:
    RIGID_BODY_2D_CLI_PATH = '/home/isaiah/scisim/build/rigidbody2dcli/rigidbody2d_cli'


def run_rigidbody2dcli_dir(
    in_dir,
    out_dir,
    exe = RIGID_BODY_2D_CLI_PATH
):
    for f in os.listdir(in_dir):
        os.system(f"{exe} -e 0.05 {in_dir}{f} > {out_dir}{f}.out")

def run_rigidbody2dcli(
    in_file,
    out_file,
    exe = RIGID_BODY_2D_CLI_PATH
):
    os.system(f"{exe} -e 0.05 {in_file} > {out_file}")


if __name__ == "__main__":
    run_rigidbody2dcli_dir(INPUT_DIR, OUT_DIR, RIGID_BODY_2D_CLI_PATH)
