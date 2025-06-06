---
layout:       post
title:        "【现代控制理论】6-Optimal Control"
author:       "Orchid"
header-style: text
catalog:      true
hidden:       false
tags:
    - 控制
    - notes
---

## 什么是最优控制

<img src="https://notes.sjtu.edu.cn/uploads/upload_722c018d688e97e90fbf7e477218c53b.png" style="zoom:67%;" />

上图是一个经典的最优控制问题。接下来，我用数学语言描述一下一般的最优控制问题：

$$
\begin{align*}
\min_{u} & \quad \mathcal{J}(F, t) \\
\text{s.t.} & \quad \dot{x} = F(x, u, t), \\
& \quad x(t_0) = x_0.
\end{align*}
$$

可以看到，最优控制问题由以下几个要素构成：

* 状态方程：受控系统的动态特性由一组一阶微分方程来描述，即状态方程。

* 目标集：通常来说，起始状态（初态）是给定的，最优控制问题往往对所达到的状态（末态）有约束。满足模态约束的状态集合称为目标集。（也可能没有末态约束）

* 容许控制：就是对控制量 $u$ 的约束。

* 性能指标：目标方程，即优化的式子。通常情况下，最优控制的性能指标如下，
  
  $$
  J = \theta \left( x(t_f), t_f \right) + \int_{t_0}^{t_f} F \left( x(t), u(t), t \right) dt
  $$
  
  第一项为末值型性能指标，与末态相关；第二项为积分型性能指标，与过程相关。

最优控制问题所求解的就是获得**最优性能指标** $J^\star$ 时所采取的**最优控制** $u^\star$ 以及**最优轨线** $x^\star(t)$。

## Preliminaries

#### 泛函与变分

* 什么是泛函

  如果对某一类函数 $\{X(t)\}$ 中的每一个函数 $X(t)$，有一个实数值 $J$ 与之相对应，则称 $J$ 为依赖于函数 $X(t)$ 的泛函，记为
  
  $$
  J = J[X(t)]
  $$
  
  粗略来说，泛函是以函数为自变量的函数（函数的函数）。很明显，最优控制问题所优化的性能指标就是一个泛函。

* 什么是变分

  * 函数的变分（等价于函数下自变量的微分）

    自变量函数 $X(t)$ 的变分 $\delta X$ 是指同属于函数类 $\{X(t)\}$ 中的两个函数 $X_1(t), X_2(t)$ 之差：
    
    $$
    \delta X = X_1(t) - X_2(t)
    $$

  * 泛函的变分（等价于函数的微分）

    当自变量函数有变分 $\delta X$ 时，泛函的增量为
    
    $$
    \Delta J = J[X + \delta X] - J[X] = L(X, \delta X) + r(X, \delta X)
    $$
    
    其中，$L(X, \delta X)$ 是 $\delta X$ 的**线性**连续泛函，称为泛函的变分，记作 $\delta J$；$r(X, \delta X)$ 是关于 $\delta X$ 的高阶无穷小。

* 泛函的变分定理
  
  $$
  \delta \mathcal{J} = \frac{\partial}{\partial \varepsilon} \mathcal{J}(x + \varepsilon \delta x) \Big|_{\varepsilon=0}
  $$
  
  （将泛函的变分等价于关于引入变量的导数）

  证明：
  
  $$
  \begin{align*}
  \delta \mathcal{J} = \frac{\partial}{\partial \varepsilon} \mathcal{J}(x + \varepsilon \delta x) \Big|_{\varepsilon=0} &=  \lim_{\varepsilon \to 0} \frac{\mathcal{J}(x + \varepsilon \delta x) - \mathcal{J}(x)}{\varepsilon} \\
  &= \lim_{\varepsilon \to 0} \frac{1}{\varepsilon} (L(x + \varepsilon \delta x) + r(x + \varepsilon \delta x)) \\
  &= \lim_{\varepsilon \to 0} \frac{1}{\varepsilon} (\varepsilon L(x +  \delta x) + r(x + \varepsilon \delta x)) \quad \text{线性} \\
  &= L(x, \delta x) + \lim_{\varepsilon \to 0} \frac{r(x + \varepsilon \delta x)}{\varepsilon \delta x} \delta x = L(x, \delta x).
  \end{align*}
  $$

* 泛函的极值定理

  * 定理：若泛函 $J (X)$ 有极值，则必有 $\delta J = 0$。

  * 证明：设泛函 $J (X)$ 在 $x_0(t)$ 处有极值。对设定的 $x_0(t)$，可将 $J(x_0 + \varepsilon \delta x)$ 看成是 $\varepsilon$ 的函数，且在 $\varepsilon = 0$ 处有极值。所以，当  $\varepsilon = 0$ 时，由极值的必要条件，导数必为零，即
    
    $$
    \delta \mathcal{J} = \frac{\partial}{\partial \varepsilon} \mathcal{J}(x + \varepsilon \delta x) \Big|_{\varepsilon=0} = 0
    $$


#### Euler Equation

考虑以下泛函：

$$
\mathcal{J} = \int_{t_0}^{T} F(\dot{x}, x, t) \mathrm{d}t
$$

自变量函数 $x(t)$ 定义在区间 $t\in[t_0,T]$ 上，$F(\dot{x}, x, t)$ 关于 $\dot{x}, x, t$  连续且有二阶连续偏导数。状态量的初值和终值已定，$x(t_0) = x_0, x(T) = x_1$，求最优轨线 $x(t)$ 使得 $\mathcal{J}$ 有极值。

$$
\begin{align}
\delta \mathcal{J} &= \int_{t_0}^{T} \left( \frac{\partial F}{\partial \dot{x}} \delta \dot{x} + \frac{\partial F}{\partial x} \delta x \right) \mathrm{d}t \\
&= \int_{t_0}^{T} \left[ \left( \frac{\partial F}{\partial x} - \frac{\mathrm{d}}{\mathrm{d}t} \cdot \frac{\partial F}{\partial \dot{x}} \right) \delta x \right] \mathrm{d}t + \left. \frac{\partial F}{\partial \dot{x}} \delta x \right|_{t_0}^{T} \quad \text{分部积分}
\end{align}
$$

由于状态量的初值和终值已定，因此 $\delta x(T) = \delta x(t_0) = 0$ 。（轨线不管怎么变，两头都是定的）因此，由极值定理：

$$
\delta \mathcal{J} = \int_{t_0}^{T} \left[ \left( \frac{\partial F}{\partial x} - \frac{\mathrm{d}}{\mathrm{d}t} \cdot \frac{\partial F}{\partial \dot{x}} \right) \delta x \right] \mathrm{d}t = 0 \\
\implies  \frac{\partial F}{\partial x} - \frac{\mathrm{d}}{\mathrm{d}t} \cdot \frac{\partial F}{\partial \dot{x}} = 0 \quad \text{(Euler Equation)}
$$

该方程为二阶常微分方程，可以得到通解。在两端固定得问题中，通解中的两个任意常数由边界条件确定。 



## 最优控制

$$
\begin{align*}
\min_{u} & \quad \mathcal{J}[u] = \varphi(x(t_1), t_1) + \int_{t_0}^{t_1} L(t, x, u) \mathrm{d}t, \\
\text{s.t.} & \quad \dot{x} = F(x, u, t), \\
& \quad x(t_0) = x_0 \in R^m.
\end{align*}
$$

* 最优控制的必要条件

  控制输入 $u^\star$ 是最优的，并且 $\mathcal{J}$ 取得极小值，当假设存在 $\varepsilon > 0$ ，对于任意的控制输入 $u$ 满足 $\|u - u^\star\| < \varepsilon$ ，均有
  
  $$
  \mathcal{J}[u] - \mathcal{J}[u^\star] \geq 0.
  $$
  
  用变分的写法来表示：
  
  $$
  \delta \mathcal{J}[u^\star, \delta u] = 0
  $$

#### 极小值定理

假设控制输入 $u$ 是不受限制的。

* 引入拉格朗日乘子 $p(t) = [p_1(t) \quad p_2(t) \quad \ldots \quad p_m(t)]$
  
  $$
  \begin{align}
  \mathcal{J}_a &= \varphi(x(t_1), t_1) + \int_{t_0}^{t_1} \left( L(t, x, u) + p(F(t, x, u) - \dot{x}) \right) \mathrm{d}t \\
  &= \varphi(x(t_1), t_1) + \int_{t_0}^{t_1} (L + pF + \dot{p}x) \mathrm{d}t - px \Big|_{t_0}^{t_1} \quad \text{分部积分} \\
  &= \varphi(x(t_1), t_1) - px \Big|_{t_0}^{t_1} + \int_{t_0}^{t_1} (H + \dot{p}x) \mathrm{d}t,
  \end{align}
  $$
  
  其中，定义 Hamiltonian function $H(t, p, x, u) := L(t, x, u) + pF(t, x, u)$ 。

* 由泛函的极值定理，做变分
  
  $$
  \begin{align}
  \delta \mathcal{J}_a &= \left[ \left( \frac{\partial \varphi}{\partial x} \delta x \right) \right]_{t=t_1} - p\delta x\Big|_{t_0}^{t_1} + \int_{t_0}^{t_1} \left( \frac{\partial H}{\partial x} \delta x + \frac{\partial H}{\partial u} \delta u + \dot{p} \delta x \right) \mathrm{d}t \\
  & = \left[ \left( \frac{\partial \varphi}{\partial x} - p \right) \delta x  \right]_{t=t_1} + \int_{t_0}^{t_1} \left( \frac{\partial H}{\partial x} \delta x + \frac{\partial H}{\partial u} \delta u + \dot{p} \delta x \right) \mathrm{d}t  = 0
  \end{align}
  $$

* 定理结论
  
  $$
  \begin{cases}
  p(t_1) = \frac{\partial \varphi}{\partial x}\Big|_{t= t_1} \\
  \dot{p} = -\frac{\partial H}{\partial x} \\
  \frac{\partial H}{\partial u}\Big|_{u= u^\star} = 0 \quad \text{(optimal control condition)}
  \end{cases} \tag{1}
  $$

#### 极小值定理的补充

* 假设泛函 $L, F$ 不依赖于时间 $t$
  
  $$
  \begin{align*}
  H(p, x, u) &= L(x, u) + pF(x, u) \\
  \implies \dot{H} &= \frac{\partial L}{\partial u} \dot{u} + \frac{\partial L}{\partial x} \dot{x} + p \left( \frac{\partial F}{\partial u} \dot{u} + \frac{\partial F}{\partial x} \dot{x} \right) + \dot{p} F \\
  &= \left( \frac{\partial L}{\partial u} + p \frac{\partial F}{\partial u} \right) \dot{u} + \left( \frac{\partial L}{\partial x} + p \frac{\partial F}{\partial x} \right) \dot{x} + \dot{p} F \\
  &= \frac{\partial H}{\partial u} \dot{u} + \frac{\partial H}{\partial x} \dot{x} + \dot{p} F \\
  &= \frac{\partial H}{\partial u} \dot{u} + \left( \frac{\partial H}{\partial x} + \dot{p} \right) F.
  \end{align*}
  $$
  
  由于在最优轨线上满足结论 $(1)$，即
  
  $$
  \dot{p} = -\frac{\partial H}{\partial x} \quad \text{and} \quad \left. \frac{\partial H}{\partial u} \right|_{u=u^*} = 0
  $$
  
  可以得到 $\dot{H} = 0$ 当控制输入 $u = u^\star$，因此 $H_{u = u^\star} = constant, t_0 \leq t \leq t_1$。

* 如果控制输入量 $u$ 是受限制的 

  假设存在最优控制 $u^\star$ ，且 $u^\star + \delta u$ 是满足约束的，对于 $\|\delta u\|$ 足够小时，则必须满足：
  
  $$
  \delta \mathcal{J}[u^\star, \delta u] \geq 0
  $$
  
  根据极小值定理的证明过程：
  
  $$
  \mathcal{J}_a = \varphi(x(t_1), t_1) - px \Big|_{t_0}^{t_1} + \int_{t_0}^{t_1} (H + \dot{p}x) \mathrm{d}t \\
  \delta \mathcal{J}_a[x, u, \delta x, \delta u] = \left[ \left( \frac{\partial \varphi}{\partial x} - p \right) \delta x  \right]_{t=t_1} + \int_{t_0}^{t_1} \left( \frac{\partial H}{\partial x} \delta x + \frac{\partial H}{\partial u} \delta u + \dot{p} \delta x \right) \mathrm{d}t
  $$
  
  且控制输入量 $u$ 受限制不影响结论 $(1)$ 中的前两条结论，因此目标泛函的变分变为：
  
  $$
  \delta \mathcal{J}_a[u, \delta u] = \int_{t_0}^{t_1} \left( H(t, p, x, u + \delta u) - H(t, p, x, u) \right) \mathrm{d}t
  $$
  
  （仅和控制量相关的项发生了变化）又因为 $\delta \mathcal{J}[u^\star, \delta u] \geq 0$，则
  
  $$
  H(t, p^*, x^*, u^* + \delta u) \geq H(t, p^*, x^*, u^*)
  $$
  
  由此得到了控制输入量 $u$ 受限制情况下的极小值定理结论：
  
  $$
  \begin{cases}
  p(t_1) = \frac{\partial \varphi}{\partial x}\Big|_{t= t_1} \\
  \dot{p} = -\frac{\partial H}{\partial x} \\
  H(t, p^*, x^*, u^* + \delta u) \geq H(t, p^*, x^*, u^*) \quad \text{for all admissible } \delta u \text{ and all t in [t0, t1]}
  \end{cases} \tag{2}
  $$

#### 极小值定理例题

* **例题一**

  已知状态方程如下：
  
  $$
  \begin{cases}
  \dot{x}_1(t) = x_2(t) \\
  \dot{x}_2(t) = u(t)
  \end{cases}
  \Rightarrow
  \begin{bmatrix}
  \dot{x}_1(t) \\
  \dot{x}_2(t)
  \end{bmatrix}
  =
  \begin{bmatrix}
  0 & 1 \\
  0 & 0
  \end{bmatrix}
  \begin{bmatrix}
  x_1(t) \\
  x_2(t)
  \end{bmatrix}
  +
  \begin{bmatrix}
  0 \\
  1
  \end{bmatrix}
  u
  \quad , \quad
  x(0) = \begin{bmatrix} 1 \\ 1 \end{bmatrix}, \quad x(2) = \begin{bmatrix} 0 \\ 0 \end{bmatrix}
  $$
  
  考虑最优控制问题
  
  $$
  \begin{align*}
  \min & \quad \mathcal{J} = \int_{0}^{2}\frac{1}{2} u^2(t) \mathrm{d}t, \\
  \text{s.t.} & \quad \dot{x}(t) = Ax + Bu
  \end{align*}
  $$
  
  引入拉格朗日乘子
  
  $$
  \mathcal{J}_a = \int_{0}^{2}\frac{1}{2} u^2(t)  + p[Ax + Bu - \dot{x}] \mathrm{d}t \\
  $$
  
  可知
  
  $$
  \begin{cases}
  \varphi = 0 \\
  H = \frac{1}{2} u^2(t) + p_1x_2 + p_2u
  \end{cases}
  $$
  
  由极小值定理结论可以得到：
  
  $$
  \begin{cases}
  \dot{p} = -\frac{\partial H}{\partial x} \implies \begin{cases} \dot{p}_1 = 0 \\ \dot{p}_2 = -p_1 \end{cases} \implies \begin{cases} p_1 = c_1 \\ p_2 = -c_1t + c_2 \end{cases}\\
  \frac{\partial H}{\partial u}\Big|_{u= u^\star} = 0 \implies u^\star = -p_2 = c_1t - c_2
  \end{cases}
  $$
  
  将最优控制代入状态方程，可以得到最优轨线：
  
  $$
  \begin{cases}
  \dot{x}_2(t) = u^\star = c_1t - c_2 \implies x_2(t) = \frac{1}{2}c_1t^2 - c_2 t + c_3 \\
  \dot{x}_1(t) = x_2(t) = \frac{1}{2}c_1t^2 - c_2 t + c_3 \implies x_1(t) = \frac{1}{6}c_1t^3 - \frac{1}{2}c_2t^2 + c_3t + c_4\\
  \end{cases}
  $$
  
  代入边界条件可以求解参数 $c_1,c_2,c_3,c_4$
  
  $$
  c_1 = 3, \quad c_2 = 3.5, \quad c_3 = c_4 = 1
  $$
  
  由此确定了最优控制和最优轨线。

* **例题二**

  已知状态方程如下：
  
  $$
  \begin{cases}
  \dot{x}_1(t) = x_2(t) \\
  \dot{x}_2(t) = u(t)
  \end{cases}
  \Rightarrow
  \begin{bmatrix}
  \dot{x}_1(t) \\
  \dot{x}_2(t)
  \end{bmatrix}
  =
  \begin{bmatrix}
  0 & 1 \\
  0 & 0
  \end{bmatrix}
  \begin{bmatrix}
  x_1(t) \\
  x_2(t)
  \end{bmatrix}
  +
  \begin{bmatrix}
  0 \\
  1
  \end{bmatrix}
  u
  \quad , \quad
  x(0) = \begin{bmatrix} 1 \\ 1 \end{bmatrix}
  $$
  
  考虑最优控制问题
  
  $$
  \begin{align*}
  \min & \quad \mathcal{J} = \int_{0}^{1}\frac{1}{2} u^2(t) \mathrm{d}t, \\
  \text{s.t.} & \quad \dot{x}(t) = Ax + Bu \\
  			& \quad x_1(1) + x_2(1) - 1 = 0
  \end{align*}
  $$
  
  引入拉格朗日乘子
  
  $$
  \mathcal{J}_a = \mu(x_1(1) + x_2(1) - 1) +  \int_{0}^{2}\frac{1}{2} u^2(t)  + p[Ax + Bu - \dot{x}] \mathrm{d}t \\
  $$
  
  可知
  
  $$
  \begin{cases}
  \varphi = \mu(x_1(1) + x_2(1) - 1) \\
  H = \frac{1}{2} u^2(t) + p_1x_2 + p_2u
  \end{cases}
  $$
  
  由极小值定理结论可以得到：
  
  $$
  \begin{cases}
  p(t_1) = \frac{\partial \varphi}{\partial x}\Big|_{t= t_1} \implies p(1) =  [\mu \quad \mu]\\
  \dot{p} = -\frac{\partial H}{\partial x} \implies \begin{cases} \dot{p}_1 = 0 \\ \dot{p}_2 = -p_1 \end{cases} \implies \begin{cases} p_1(t) = c_1 \\ p_2(t) = -c_1t + c_2 \end{cases} \implies \begin{cases} p_1(t) = \mu \\ p_2(t) = -\mu t + 2\mu \end{cases} \\
  \frac{\partial H}{\partial u}\Big|_{u= u^\star} = 0 \implies u^\star = -p_2 = \mu t - 2\mu
  \end{cases}
  $$
  
  将最优控制代入状态方程，可以得到最优轨线：
  
  $$
  \begin{cases}
  \dot{x}_2(t) = u^\star = \mu t - 2\mu \implies x_2(t) = \frac{1}{2}\mu t^2 - 2\mu t + c_3 \\
  \dot{x}_1(t) = x_2(t) = \frac{1}{2}c_1t^2 - c_2 t + c_3 \implies x_1(t) = \frac{1}{6}\mu t^3 - \mu t^2 + c_3t + c_4\\
  \end{cases}
  $$
  
  代入边界条件可以求解参数 $c_1,c_2,c_3,c_4$
  
  $$
  \mu = \frac{6}{7}, \quad c_3 = c_4 = 1
  $$
  
  由此确定了最优控制和最优轨线。



## Quadratic optimal control system

考虑以下状态方程和性能指标：

$$
\begin{align}
&\begin{cases}
\dot{x} = Ax + Bu \\
y = Cx
\end{cases} \\
\mathcal{J} &:= \int_{0}^{\infty} \frac{1}{2}(x^\top Q x + u^\top R u) \mathrm{d}t
\end{align}
$$

其中，$Q$ 和 $R$ 是设计者选定的权重矩阵（对称正定阵）。

* 目标：设计最优控制 $u^\star$ ，使得性能指标 $\mathcal{J}$ 取得最小值。同时，因为是针对开环状态方程，我们考虑使用**状态反馈控制**。

对于二次型最优控制问题，有两种解题思路：

* **思路一：**

  先从最优控制问题的角度出发，引入拉格朗日乘子：
  
  $$
  \mathcal{J}_a := \int_{0}^{\infty} \frac{1}{2}(x^\top Q x + u^\top R u) + p(Ax + Bu - \dot{x})\mathrm{d}t
  $$
  
  可知
  
  $$
  \begin{cases}
  \varphi = 0 \\
  H = \frac{1}{2}(x^\top Q x + u^\top R u) + p(Ax + Bu)
  \end{cases}
  $$
  
  由极小值定理结论 $(1)$ 可以得到
  
  $$
  \text{（我自己的推导）}
  \begin{cases}
  p(t_1) = \frac{\partial \varphi}{\partial x}\Big|_{t= t_1} \implies p(\infty) = 0 \\
  \dot{p} = -\frac{\partial H}{\partial x} \implies \dot{p} = -(x^TQ + pA)\\
  \frac{\partial H}{\partial u}\Big|_{u= u^\star} = 0 \implies {u^\star}^T R + pB = 0 \implies u^\star = -R^{-1}B^Tp^T
  \end{cases}
  $$

  $$
  \text{（老师上课推导原式，感觉有问题）}
  \begin{cases}
  p(t_1) = \frac{\partial \varphi}{\partial x}\Big|_{t= t_1} \implies p(\infty) = 0 \\
  \dot{p} = -\frac{\partial H}{\partial x} \implies \dot{p} = -(Qx + A^Tp)\\
  \frac{\partial H}{\partial u}\Big|_{u= u^\star} = 0 \implies Ru^\star + B^Tp = 0 \implies u^\star = -R^{-1} B^Tp
  \end{cases}
  $$

  再考虑状态反馈控制，即令 $p = K'x, K'\in R^{m\times m}$，则有 $u^\star = -R^{-1} B^TK'x$，令 $K = R^{-1} B^TK'$。则新的状态方程为：
  
  $$
  \dot{x} = (A - BR^{-1} B^TK')x
  $$
  
  同时，考虑式子 $\dot{p} = -(Qx + A^Tp)$，可以得到：
  
  $$
  K'\dot{x} = -(Q + A^TK')x
  $$
  
  结合上述两条式子，可以得到：
  
  $$
  K'A - K'BR^{-1} B^TK' + Q + A^TK' = 0
  $$
  
  这条式子被称为 **Riccati equation**，式子中仅有 $K'$ 是未知的，因此可以通过等式求解（通常是要求对称正定解）。至此最优控制和最优轨线都已经能够得到。

  还需要考虑的事情是，在上述求得的最优控制下，系统是否是李雅普诺夫稳定的？令 $V(x) = x^TK'x$ ，则有：
  
  $$
  \begin{align}
  \frac{dV(x)}{dt} &= \dot{x}^TK'x + x^TK'\dot{x} \\
                   &= x^T(A - BR^{-1} B^TK')^TK'x + x^TK'(A - BR^{-1} B^TK')x \\
                   &= x^T(A^TK' - K'^TBR^{-T}B^TK' + K'A - K'BR^{-1} B^TK')x \quad \text{代入 Riccati equation}\\
                   &= x^T(- K'^TBR^{-T}B^TK' - Q)x
  
  \end{align}
  $$
  
  因为 $R$ 和 $Q$ 是对称正定阵，因此 $\frac{dV(x)}{dt}$ 是负定的。因此在最优控制下，系统是李雅普诺夫渐进稳定的。



* **思路二：**

  直接从状态反馈的角度出发，令 $u = -Kx$，则 $\dot{x} = (A-BK)x$。令 $V(x) = x^TPx$ ，则有：
  
  $$
  \begin{align}
  \frac{dV(x)}{dt} &= \dot{x}^TPx + x^TP\dot{x} \\
                   &= x^T[P(A-BK) + (A-BK)^TP]x
  \end{align}
  $$
  
  为了保证系统是渐近稳定的，上述式子应当要保证负定。我们先假设是负定的。

  考虑最优控制的性能指标：
  
  $$
  \begin{align*}
  \mathcal{J} &= \int_{0}^{\infty} (x^\top Q x + u^\top R u) \mathrm{d}t \\
  &= \int_{0}^{\infty} \left( x^\top Q x + u^\top R u + \frac{\mathrm{d}V(x)}{\mathrm{d}t} \right) \mathrm{d}t - \int_{0}^{\infty} \frac{\mathrm{d}V(x)}{\mathrm{d}t} \mathrm{d}t \\
  &= \int_{0}^{\infty} \left( x^\top Q x + u^\top R u + x^\top [P(A - BK) + (A - BK)^\top P] x \right) \mathrm{d}t - V[x(t)] \Big|_{t=0}^{t=\infty} \\
  &= \int_{0}^{\infty} x^\top \left[ Q + K^\top R K + PA + A^\top P - PBK - K^\top B^\top P \right] x \mathrm{d}t + x_0^\top P x_0
  \end{align*}
  $$

  我们的目标是选择 $K$ 使得性能指标 $\mathcal{J}$ 取到最小值。可以发现:
  
  $$
  \begin{align*}
  K^\top RK - PBK - K^\top B^\top P &= K^\top RK - PBK - K^\top B^\top P + PBR^{-1}B^\top P - PBR^{-1}B^\top P \\
  &= (K - R^{-1}B^\top P)^\top R(K - R^{-1}B^\top P) - PBR^{-1}B^\top P
  \end{align*}
  $$
  
  将上式代入性能指标中，可以得到：
  
  $$
  \begin{align*}
  \mathcal{J} &= \int_{0}^{\infty} x^\top [Q  + PA + A^\top P + (K^\top RK - PBK - K^\top B^\top P)] x \mathrm{d}t + x_0^\top Px_0 \\
  &= \int_{0}^{\infty} x^\top [Q + PA + A^\top P - PBR^{-1}B^\top P] x \mathrm{d}t + x_0^\top Px_0  + \int_{0}^{\infty} x^\top (K - R^{-1}B^\top P)^\top R(K - R^{-1}B^\top P) x \mathrm{d}t
  \end{align*}
  $$
  
  因此，选择 $K = R^{-1}B^\top P$，选择 $P$ 满足 $Q + PA + A^\top P - PBR^{-1}B^\top P = 0$，性能指标 $\mathcal{J}$ 即可取到最小值 $x_0^\top Px_0$ 。

  同样按照思路一后半部分稳定性证明，可以证明 $\frac{dV(x)}{dt}$ 是负定的，即假设是成立的。

