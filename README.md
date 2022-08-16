## mpsplines: Mean preserving interpolation with splines```

### Installation:

```bash
python3 -m pip install git+https://github.com/jararias/mpsplines.git
```

### Usage:

The usage of ```mpsplines``` is better explained with an example.

```python
import numpy as np
import pylab as pl

# unknown process
x = np.linspace(0., 2*np.pi, 480)
y = (0.05 - np.sin(x))**2

# we only know some averages
xi = np.reshape(x, (480, 60)).mean(axis=1)
yi = np.reshape(y, (480, 60)).mean(axis=1)
```

where ```y``` is the unknown process that we want to reconstruct, but we only know the averages:

![Figure 1](https://github.com/jararias/mpsplines/assets/figure_01.png)

