from pathlib import Path
from setuptools import setup, find_packages

name = "mpsplines"
version = "0.1"

version_file = Path(f'{name}/_version.py')
with open(version_file, 'w') as f:
    f.write(f'__version__ = "{version}"')


def read_content(fname):
    absolute_fname = Path(__file__).parent.joinpath(fname)
    with open(absolute_fname, mode="r", encoding="utf-8") as f:
        return f.read()


setup(
    name=name,
    version=version,
    author="Jose A. Ruiz-Arias",
    author_email="jararias@uma.es",
    url="https://github.com/jararias/msplines",
    description="Mean preserving interpolation of time series with splines",
    long_description=read_content("README.md"),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    classifiers=[
        "Natural Language :: English",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python ::3.9",
        "Development Status :: 3 - Alpha"
    ],
    python_requires=">=3.6",
    install_requires=['numpy', 'scipy', 'loguru'],
    extras_require={},
    entry_points={}
)

version_file.unlink()
