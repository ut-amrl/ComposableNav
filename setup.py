import setuptools

# Read in the requirements.txt file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Define the setup configuration
setuptools.setup(
    name="composablenav",
    version="0.1.0",
    author="Zichao Hu",
    author_email="zichao@utexas.edu",
    description="Instruction-Following Navigation in Dynamic Environments via Composable Diffusion",
    packages=setuptools.find_packages(include=["composablenav"]),
    install_requires=requirements,
    python_requires='>=3.10',
)
