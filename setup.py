import io
import os
import re

import setuptools


def get_long_description():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    with io.open(os.path.join(base_dir, "README.md"), encoding="utf-8") as f:
        return f.read()


def get_requirements():
    with open("requirements.txt") as f:
        return f.read().splitlines()


def get_version():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    version_file = os.path.join(current_dir, "pybboxes", "__init__.py")
    with io.open(version_file, encoding="utf-8") as f:
        return re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', f.read(), re.M).group(1)


_DEV_REQUIREMENTS = [
    "black==22.3.0.",
    "click==8.1.3",
    "deepdiff==5.5.0",
    "flake8==3.9.2",
    "isort==5.9.2",
    "pytest>=7.0.1",
    "pytest-cov>=3.0.0",
    "pytest-timeout>=2.1.0",
]

extras = {
    "dev": _DEV_REQUIREMENTS,
}


setuptools.setup(
    name="pybboxes",
    version=get_version(),
    author="Devrim Cavusoglu",
    license="MIT",
    description="Light Weight Toolkit for Bounding Boxes",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/devrimcavusoglu/pybboxes",
    packages=setuptools.find_packages(exclude=["tests"]),
    python_requires=">=3.7",
    install_requires=get_requirements(),
    extras_require=extras,
    include_package_data=True,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Education",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="machine-learning, deep-learning, image-processing, pytorch, tensorflow, numpy, bounding-box, iou, "
    "computer-vision, cv",
)
