#### I used deeponet-core and AAA_unsteady_Deeponet for creating this example

### Parametric 1-D Linear Advection Problem u(x,t), f(x,t)

#### PDE  

$$ u_t + c\,u_x = 0, \qquad 0 < x < L,\; 0 < t < 1. $$


#### Given data 

* Domain length $L$ may be fixed (say $L=1$), or made as parameter.

* Wave speed $c$ may be fixed (say $c = 1$ or find other value in $c\in[0.5,2.0)$), or made as parameter.

* **Inflow boundary condition** (BC, because \(c>0\)). Parameter $k$ – forcing signal: the entire time-series $g(t)$; this is what the branch net must encode.
  $$ u(0,t)=g(t), \qquad 0\le t\le 1.$$
Example
$$g(t) = \sin\!\Bigl(\tfrac{\pi k}{2}\,t\Bigr) + \tfrac{\pi kt}{2}$$

* **Initial condition** (IC)  
  $$ u(x,0)=\sin(\pi x), \qquad 0\le x\le L. $$
* **No condition at the outflow end** \(x=L\).

---

### Closed-form solution  

Trace characteristics $(x-c(t-\tau))$:

$$ u(x,t;c)=
\begin{cases}
\displaystyle \sin\!\bigl[\pi\,(x-c\,t)\bigr], & \text{if }\;x\ge c\,t, \\[8pt]
\displaystyle \sin\!\Bigl[\dfrac{\pi k}{2}\,\bigl(t-\dfrac{x}{c}\bigr)\Bigr] + \dfrac{\pi k (t-\dfrac{x}{c})}{2}, & \text{if }\;0\le x< c\,t.
\end{cases}
$$

*The dividing line $(x=c\,t)$ separates points that inherit the initial data (first branch) from those that inherit the boundary data (second branch).  Both branches agree at $(x=c\,t)$ because $\sin 0 = 0$, so the solution is continuous throughout the domain.*
