from numpy import *
from scipy.integrate import *

def fun(t, xy):
    x, y = xy
    return [-y, x]

y0s = [[cos(th), sin(th)] for th in linspace(0, 2*pi, 10000)]

t_eval = linspace(0.0, 10.0, 10000)
t_span = [0.0, 10.0]
atol=1e-15

data = []
for y0 in y0s:
    data.append(
        solve_ivp(fun=fun, y0=y0, t_span=t_span, t_eval=t_eval, atol=atol).y
    )
print(data)



