from setuptools import setup, find_packages

version = "0.1"

setup(
    name="mpsplines",
    version=version,
    author="Jose A. Ruiz-Arias",
    author_email="jararias@uma.es",
    url="",
    description="",
    packages=find_packages(),
    classifiers=[],
    python_requires=">=3.6",
    install_requires=['numpy', 'scipy', 'loguru'],
    entry_points={}
)
