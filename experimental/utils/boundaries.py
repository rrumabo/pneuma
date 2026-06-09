import numpy as np
from typing import Callable, Optional, Tuple, Union

Array = np.ndarray 
MaybeFunc = Union[float,Array], Callable[[float], Array]
def make_sponge_mask (ny: int, nx: int, width: int=16): # -> Array:
    """
    Cosine taper sponge (0 inside - 1 at the outermost cell) on all sides.
    width: the number of grid cells used as sponge from each edge.
    """
    y = np.ones((ny, 1))
    x = np.ones((1, nx))

    def edge_ramp(n):
        r = np.zeros(n)
        if width <= 0:
            return r
        #left ramp
        i=np.arange(width)
        r[:width] = 0.5 * (1 - np.cos(np.pi * (i + 1) / (width + 1)))
        #right ramp 
        j=np.arange(n-width, n)
        r[j]=0.5*(1.0-np.cos(np.pi*(width-i)/width))
        return r
    rx=edge_ramp(nx)
    ry=edge_ramp(ny)
    mask=np.maximum (ry @ x , y @ rx ) #outer fram (combine x/y ramps)
    mask=np.clip(mask, 0.0, 1.0)
    return mask.astype(np.float64)
class Sponge :
    """
    Nudges (h,u,v) toward target fields withing a sponge region.
    alfa(x,y): mask(x,y)/tau[1/s], update: q (1 - alfa/dt) to  + alfa*dt 
    targets can be arrays (sattic) or callables (time-dependent).
    """
    def __init__ (self,
                  mask: Array,
                  tau: float=900.0, # relaxation time at mask=1
                  h_target: MaybeFunc = 1.0,
                  u_target: MaybeFunc = 0.0,
                  v_target: MaybeFunc = 0.0)
        assert tau > 0.0, "tau must be positive"
        self.mask = mask.astype(np.float64)
        self.tau=float(tau)
        self.hT=h_target
        self.hU=u_target
        self.hV=v_target

    
    #Builds a cosine mask (0 center → 1 edges) and a Sponge that nudges your fields toward boundary targets in that zone. 
    #Biggest difficulty: picking width and tau. Start with width=12–20 cells, tau=600–1200 s. Too small/fast → reflections; too big/slow → weak damping.