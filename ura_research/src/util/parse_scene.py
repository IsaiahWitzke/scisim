import argparse
from dataclasses import dataclass
from typing import List
import numpy as np
import xml.etree.ElementTree as ET
from math import pi

@dataclass
class SimulationParams:
    generalized_normal_basis: np.array  # aka "N", the normal vectors of each collision
    quadratic_term: np.array            # aka "Q"
    coefficient_of_restitution: float   # aka "CoR"
    ball_masses_inv: np.array           # aka "M ^-1"
    ball_velocities: np.array           # aka "q dot" or "v0", the velocity of the balls pre-collision
    b: np.array                         # the constant term of the quadratic equation, b = N (q dot) (1 + c_r)
                                        # however, we generally assume collisions are perfectly elastic, so b = N (q dot) (2)


@dataclass
class Ball:
    pos: np.array
    velocity: np.array
    mass: float
    radius: float


@dataclass
class Collision:
    ball_1: Ball
    ball_2: Ball
    ball_1_idx: int
    ball_2_idx: int


def parse_ball_data(
    file_path: str,
) -> (List[Ball], float):
    """
    Returns list of positions, velocities, and masses of the balls present in the xml
    """
    xml_tree = ET.parse(file_path)
    xml_root = xml_tree.getroot()
    xml_balls = xml_root.findall("rigid_body")

    balls: List[Ball] = []

    for ball in xml_balls:
        pos_str = ball.get("x")
        v_str = ball.get("v")
        r = float(ball.get("r"))
        density = float(ball.get("rho"))
        balls.append(
            Ball(
                pos=np.array([float(pos_str.split()[0]), float(pos_str.split()[1])]),
                velocity=np.array([float(v_str.split()[0]), float(v_str.split()[1])]),
                mass=pi * (r**2) * density,
                radius=r,
            )
        )
    
    coefficient_of_restitution = float(xml_root.find("impact_operator").get("CoR"))

    return balls, coefficient_of_restitution 


def parse_scene(file_path) -> SimulationParams:
    balls, coefficient_of_restitution = parse_ball_data(file_path)
    collisions: List[Collision] = []

    for ball_1_idx in range(len(balls)):
        for ball_2_idx in range(ball_1_idx):
            ball_1 = balls[ball_1_idx]
            ball_2 = balls[ball_2_idx]
            if np.linalg.norm(ball_1.pos - ball_2.pos) < ball_1.radius + ball_2.radius:
                collisions.append(
                    Collision(
                        ball_1=ball_1,
                        ball_2=ball_2,
                        ball_1_idx=ball_1_idx,
                        ball_2_idx=ball_2_idx,
                    )
                )

    generalized_normal_basis = np.zeros((3 * len(balls), len(collisions)))
    for collision_idx, collision in enumerate(collisions):
        # gradient of the "g" function (i.e. the constraint function) happens to be the 
        # direction ball 1 is from ball 2 (from ball 1's PoV, opposite for ball 2)
        ball_1_2_direction = collision.ball_1.pos - collision.ball_2.pos
        # normalize
        ball_1_2_direction = ball_1_2_direction / np.linalg.norm(ball_1_2_direction)

        generalized_normal_basis[collision.ball_1_idx * 3, collision_idx] = ball_1_2_direction[0]
        generalized_normal_basis[collision.ball_1_idx * 3 + 1, collision_idx] = ball_1_2_direction[1]

        generalized_normal_basis[collision.ball_2_idx * 3, collision_idx] = -ball_1_2_direction[0]
        generalized_normal_basis[collision.ball_2_idx * 3 + 1, collision_idx] = -ball_1_2_direction[1]
    
    # we assume that the mass of each ball is 1
    ball_masses_inverse = np.eye(3 * len(balls))

    # for some reason, every 3rd entry down the Minv diagonal is "2" in the c++ scisim...
    # idt it actually matters (since rotational velocity == 0 all the time),
    # but I'll be consistent with the c++ implementation of scisim here
    for i in range(len(balls)):
        ball_masses_inverse[3 * i + 2, 3 * i + 2] = 2.0

    quadratic_term = generalized_normal_basis.T @ ball_masses_inverse @ generalized_normal_basis

    ball_velocities = np.zeros((3 * len(balls), 1))

    for ball_idx, ball in enumerate(balls):
        ball_velocities[3 * ball_idx, 0] = ball.velocity[0]
        ball_velocities[3 * ball_idx + 1, 0] = ball.velocity[1]
    
    b = (1 + coefficient_of_restitution) * generalized_normal_basis.T @ ball_velocities

    return SimulationParams(
        generalized_normal_basis=generalized_normal_basis,
        quadratic_term=quadratic_term,
        coefficient_of_restitution=coefficient_of_restitution,
        ball_masses_inv=ball_masses_inverse,
        ball_velocities=ball_velocities,
        b=b
    )
    


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("file_path", help="the path to the .xml 'scene' file you wish to parse")
#     args = parser.parse_args()
#     parse_scene(args.file_path)
parse_scene("/home/isaiah/scisim/research/scenes/simple/3_balls_1_collision.xml")
