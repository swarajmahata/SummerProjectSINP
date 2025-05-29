from typing import List, Callable
from scipy.interpolate import interp1d

class InterpolationMethod:
    def interpolate(self, x: List[float], y: List[float]) -> Callable[[float], float]:
        raise NotImplementedError("Subclasses should implement this method.")

class LinearInterpolator(InterpolationMethod):
    def interpolate(self, x: List[float], y: List[float]) -> Callable[[float], float]:
        return interp1d(x, y, kind='linear', fill_value="extrapolate")

class CubicInterpolator(InterpolationMethod):
    def interpolate(self, x: List[float], y: List[float]) -> Callable[[float], float]:
        return interp1d(x, y, kind='cubic', fill_value="extrapolate")

class Interpolator:
    def __init__(self, method: InterpolationMethod):
        self.method = method

    def interpolate(self, x: List[float], y: List[float]) -> Callable[[float], float]:
        return self.method.interpolate(x, y)
