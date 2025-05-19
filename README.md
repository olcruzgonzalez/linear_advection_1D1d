#### I used deeponet-core and AAA_unsteady_Deeponet for creating this example

### Parametric 1-D Linear Advection Problem

#### PDE  

$$ u_t + c\,u_x = 0, \qquad 0 < x < 1,\; 0 < t < 1,\qquad c>0\;(\text{parameter}). $$


#### Given data  
* **Initial condition** (IC)  
  $$ u(x,0)=\sin(\pi x), \qquad 0\le x\le 1. $$
* **Inflow boundary condition** (BC, because \(c>0\))  
  $$ u(0,t)=\sin\!\Bigl(\tfrac{\pi}{2}\,t\Bigr), \qquad 0\le t\le 1.$$
* **No condition at the outflow end** \(x=1\).

---

### Closed-form solution  

Trace characteristics $(x-c(t-\tau))$:

$$ u(x,t;c)=
\begin{cases}
\displaystyle \sin\!\bigl[\pi\,(x-c\,t)\bigr], & \text{if }\;x\ge c\,t, \\[8pt]
\displaystyle \sin\!\Bigl[\dfrac{\pi}{2}\,\bigl(t-\dfrac{x}{c}\bigr)\Bigr], & \text{if }\;0\le x< c\,t.
\end{cases}
$$

*The dividing line $(x=c\,t)$ separates points that inherit the initial data (first branch) from those that inherit the boundary data (second branch).  Both branches agree at $(x=c\,t)$ because $\sin 0 = 0$, so the solution is continuous throughout the domain.*
