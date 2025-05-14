# I used deeponet-core and AAA_unsteady_Deeponet for creating this example

### A “small-but-complete” parametric PDE

**Linear advection (transport) equation**

$$
[PHY] \quad \frac{\partial u}{\partial t}+c\,\frac{\partial u}{\partial x}=0,
\qquad x\in[0,1],\;t\in [0,1]
$$


$$
[IC] \quad u(x,0;c)=g(x) = sin(x).
$$


---

#### Closed-form solution


$$
u(x,t;c)=g\!\bigl(x-ct\bigr) = sin\!\bigl(x-ct\bigr) = sin\!\bigl(x\bigr)cos\!\bigl(ct\bigr) - cos\!\bigl(x\bigr)sin\!\bigl(ct\bigr).
$$
