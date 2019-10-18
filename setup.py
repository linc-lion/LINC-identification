from setuptools import find_packages, setup

setup(
    name="linc_identification",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "fastai",
        "Pillow",
        "matplotlib",
        "numpy",
        "scikit-learn",
        "point-cloud-utils",
        "opencv-python",
        "pycpd",
    ],
)
