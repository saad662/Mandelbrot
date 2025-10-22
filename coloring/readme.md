# Mandelbrot Coloring Interface

This project includes a plugin-based coloring interface that supports various post-processing effects for visualizing the Mandelbrot set, such as:

- **Log Coloring**
- **Sqrt Coloring**
- **Histogram Equalization**

---

## 📁 Directory Structure

mandelbrot/ \
├── coloring/ \
│ ├── base.py # Contains ColoringInterface definition\
│ ├── log_coloring.py # Logarithmic post-processing\
│ ├── sqrt_coloring.py # Square-root post-processing\
│ ├── hist_eq.py # Histogram equalization\


---

## 🧩 Interface: `ColoringInterface`

All post-processing plugins must inherit from `ColoringInterface` and implement the following method:

```python
class ColoringInterface:
    def apply(self, data, Z, mask, iterations, cycles):
        pass
```

```data```: Iteration data (2D array)\
```Z```: Final complex values after iteration\
```mask```: Boolean mask of diverged points\
```iterations```: Max iteration count\
```cycles```: Number of color cycles (used for modular coloring)\

## Adding a New Coloring Plugin

Create a new file in the coloring/ folder,  e.g. "my_coloring.py"
Inherit from **ColoringInterface**
Implement the **apply()** method:

```python
from .base import ColoringInterface

class MyColoring(ColoringInterface):
    def apply(self, data, Z, mask, iterations, cycles):
        # custom logic here
        return result_array
```
## Example using LogColoring

```python
import numpy as np
from coloring.base import ColoringInterface


class LogColoring(ColoringInterface):
    def apply(self, data, Z, mask, iterations, cycles):
        t = np.abs(Z)
        arg = mask * t + (1 - mask) * 2

        arg = np.clip(arg, 1e-10, None)

        tmp = data + 1 + 1.0 / np.log(2.0) * np.log(np.log(2.0) / np.log(arg))
        return tmp % (iterations / cycles)
```

Similarly there are total 3 coloring plugin implementation available in the project named as "SqrtColoring", "HistEquilization" and "LogColoring".

Each of the implementation uses different techniques to modify the fractal numpy array and returns the modified numpy array.
## Usage
Use **load_coloring_plugin(path:str)**
function from **plugin_loader.py** to use your implemented plugin

example:
```python
plugin = load_coloring_plugin("coloring.log_coloring.LogColoring")
result = plugin.apply(data, Z, mask, iterations, cycles)

```