"""Setup file."""

from setuptools import setup

setup(
    name="gym_missile_command",
    version="2.0",
    author="NAT_PET",
    install_requires=["gym",
                      "numpy",
                      "opencv-python",
                      "pygame"],
    description="Gym environment of game, Missile Command.",
)
