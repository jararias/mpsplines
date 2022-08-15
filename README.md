### Mean-preserving interpolation with splines for averaged data series in solar radiation modeling

This repository contains the data and code prepared while developing and writing the following research article:

Ruiz-Arias, JA. "Mean-preserving interpolation with splines for averaged data series in solar radiation modeling" Submitted for publication in Solar Energy, August 2022

### Usage:

Clone the repository to the local disk from the web interface, or from a command line interface:

```bash
git clone --single-branch --branch paper https://github.com/jararias/mpsplines/tree/paper
```

Ensure that your Python environment has the packages listed in ```requirements.txt```.

Reproduce the article's figures runing the scripts ```figure*.py```.

The interpolation code is in the Python module ```mpsplines.py```.

### mpsplines package:

The mean-preserving interpolation code can be more conveniently installed in your Python environment as follows:

```python
python3 -m pip install git+https://github.com/jararias/mpsplines.git
```

This will give you access to ```mpsplines``` from your Python environment anywhere. See details in [https://github.com/jararias/mpsplines](https://github.com/jararias/mpsplines).
