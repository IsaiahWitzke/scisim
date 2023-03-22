
# Automatically generates a case similar to that of balls_in_box.xml but with a much larger number of balls.
import math
import random
import os
import sys

if len(sys.argv) == 2:
    OUTPUT_DIR = sys.argv[1]
else:
    print("Usage: python generate_balls.py <path to output folder>")
    exit()


class Ball:
    def __init__(self, q, r, v):
        self.q = q
        self.r = r
        self.v = v

    def to_xml(self):
        return f'  <rigid_body x="{self.q[0]} {self.q[1]}" theta="0" v="{self.v[0]} {self.v[1]}" omega="0" rho="{1 / math.pi}" r="{self.r}" geo_idx="0"/>'

def create_xml(grid_size, name, r = 1, spacing=1.0):

    # ball_sep = plane_size / grid_size * 2 * math.sqrt(2)
    balls = []
    s2 = math.sqrt(2)
    for x_coord in range(grid_size):
        for y_coord in range(grid_size):
            pert_factor = 0.01
            balls.append(
                Ball(
                    (spacing*x_coord*r + pert_factor*random.random(), spacing*y_coord*r + pert_factor*random.random()),
                    r,
                    (random.random() * 2 - 1, random.random() * 2 - 1)
                )
              )
    # balls = [Ball(-60,0,0.5)]

    balls_xml = '\n'.join([ball.to_xml() for ball in balls])

    # planes_xml = \
    #     f"""
    #   <static_plane x="-{plane_size} -{plane_size}" n="0.70710678118654752440 0.70710678118654752440"/>
    #   <static_plane x="{plane_size} -{plane_size}" n="-0.70710678118654752440 0.70710678118654752440"/>
    #   <static_plane x="{plane_size} {plane_size}" n="-0.70710678118654752440 -0.70710678118654752440"/>
    #   <static_plane x="-{plane_size} {plane_size}" n="0.70710678118654752440 -0.70710678118654752440"/>
    # """

    result = \
        f"""
    <rigidbody2d_scene>
    
      <camera center="0 0" scale="8.3225" fps="50" render_at_fps="0" locked="0"/>
    
      <integrator type="symplectic_euler" dt="0.01"/>
    
      <impact_operator type="gr" CoR="1.0" v_tol="1.0e-12" cache_impulses="0">
        <solver name="lcp_solver_isaiah_debug" linear_solvers="ma27 ma86" max_iters="100" tol="1.0e-12"/>
      </impact_operator>
    
      <near_earth_gravity f="0.0 0.0"/>
    
    
      <geometry type="circle" r="{1}"/>
    
    {balls_xml}
    
    </rigidbody2d_scene>
    """

    with open(os.path.join(OUTPUT_DIR,f'{name}.xml'), 'w') as f:
        f.write(result)

# sizes = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 40000, 90000, 160000]
# sizes = [3, 4, 5, 10, 20, 50]
# for itr in range(500):
#     create_xml(2, f"itr_{itr}")

def generate_ok_simulations():
    for itr in range(500):
        create_xml(3, f"ok_itr_{itr}")

if __name__ == "__main__":
    generate_ok_simulations()