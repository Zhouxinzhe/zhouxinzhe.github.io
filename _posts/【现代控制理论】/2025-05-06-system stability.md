---
layout:       post
title:        "【现代控制理论】- System Stability"
author:       "Orchid"
header-style: text
catalog:      true
hidden:       false
tags:
    - 控制
    - notes
---

在这一章中，我们将介绍系统稳定性的分析方法。相对于自动控制原理中，对于传递函数的稳定性分析，这一章将针对状态空间模型的系统建模方式，给出相应的更加全面的系统稳定性分析方法。

* **Stability**

  事实上，稳定性是有不同的定义的，或者说不同层次的稳定性：

  * **External Stability：**外部稳定，仅关注系统的**输入和输出之间的关系**，通常所说的 BIBO 稳定（传递函数），是外部稳定。
  * **Internal Stability：**内部稳定，关注系统中的**每一个状态量**，是系统状态的稳定性，相对于外部稳定，内部稳定的要求是更高的

* **定理：**一个 SISO 系统是 BIBO 稳定的，当且仅当该系统闭环传递函数的极点的实部均小于等于 0
* **定理：**一个线性系统是内部稳定的，当且仅当该系统中任意两点间的传递函数是 BIBO 稳定的。

* 外部稳定但是内部不稳定的例子：

    $$
    \begin{aligned}
    \dot{x} &= \begin{bmatrix} -1 & 0 \\ 0 & 2.5 \end{bmatrix} x + \begin{bmatrix} 1 \\ 0 \end{bmatrix} u \\
    y &= \begin{bmatrix} 1 & 1 \end{bmatrix} x
    \end{aligned}
    $$

    $$
    g(s) = \boldsymbol{c}(s\boldsymbol{I} - \boldsymbol{A})^{-1}\boldsymbol{b} = \frac{s - 2.5}{(s + 1)(s - 2.5)} = \frac{1}{s + 1}
    $$

    很明显，从传递函数看是稳定的（外部稳定），但是状态量 $x_2$ 是不收敛的，因此不是内部稳定的



* 需要注意的是

  * 外部稳定所关注的问题是，当系统的输入是有界时，系统的输出是否有界。

  * **内部稳定**所关注的问题是，**当系统状态因为扰动从稳态离开后，能否回到稳态或是维持在稳态附近**。实际上研究的是**系统稳态的稳定性**。因此研究的其实是系统零输入响应的稳定性：
    
    $$
    \dot{x} = f(x, t) \quad x(t_0) = x_0, \, t \geq t_0
    $$



## 李雅普诺夫稳定

考虑**自治系统**

$$
\dot{x} = f(x, t) \quad x(t_0) = x_0, \, t \geq t_0
$$

我们称系统状态的运动 $x(t)$ 为扰动运动（perturbed motion），即 $x(t) = \phi(t; x_0, t_0)$

* **稳态：**

  $x_e$ 是该自治系统的稳态或者是稳定点，当且仅当对于任意时刻 $t$，均有
  
  $$
  f(x_e, t) \equiv 0
  $$

  * 对于 LTI 系统，如果 A 可逆，则稳态唯一；如果 A 不可逆，则稳态无穷多（基础解系）
  * 对于非线性系统，系统可能有唯一稳态、可能有多个稳态、可能没有稳态
  * **对于任意孤立的稳态点，我们都能通过适当的坐标变换，将稳态点转移到原点位置**。本文接下来考虑的所有稳定性问题都指的是原点稳定性问题。

* **稳定（Stable）：**

  数学表述：对于任意ε>0，存在δ(ε，t₀)>0，当\|x₀\|<δ时，有\|φ(t；x₀，t₀)\|<ε，对所有t≥t₀成立

  人话：系统受到小的初始扰动后，其状态轨迹能够一直保持在某个小范围内，不会偏离平衡点太远。

  <img src="https://notes.sjtu.edu.cn/uploads/upload_695416bf8cfa0aa499b572d57d819d34.png" style="zoom:67%;" />

* **一致稳定（Uniformly Stable）：**

  数学表述：在满足李雅普诺夫稳定（Stable）的基础上，如果δ(ε，t₀)的选择与t₀无关，则为一致稳定

  人话：任意时刻的受扰运动，都是李雅普诺夫稳定的。

* **渐近稳定（Asymptotically Stable）：**

  数学表述：在满足李雅普诺夫稳定（Stable）的基础上，当t→∞时，有φ(t；x₀，t₀)→0

  人话：系统不仅能够保持在平衡点附近，而且最终能够回到平衡点。

  ​                                                         <img src="https://notes.sjtu.edu.cn/uploads/upload_93c1d05c0ba193ee642da842721c3f91.png" style="zoom:67%;" /> <img src="https://notes.sjtu.edu.cn/uploads/upload_4d532ca40a6b898fb6c7506914b915d3.png" style="zoom:40%;" />

* **不稳定（Unstable）：**

  数学表述：稳定的否命题

  人话：系统受到一些小的初始扰动后，其状态轨迹会趋于发散，逐渐远离平衡点

  <img src="https://notes.sjtu.edu.cn/uploads/upload_1cfde3a4a83e29730b7a1d548cb4ad22.png" style="zoom: 67%;" />

* **大范围渐近稳定（globally asymptotically）：**

  数学表述：对于任意初始条件x₀，都有limₜ→∞φ(t；x₀，t₀)=0

  人话：无论系统从状态空间中的哪个点出发，最终都会收敛到平衡点

  * 必要条件：状态空间中只有一个稳定点
  * **对于线性系统而言，若其稳定点是渐进稳定的，那么它必然是大范围渐近稳定的**。



## 稳定性判据

连续时间系统（自治系统）：

* **LTI 系统 $\dot{x} = Ax$ 是内部稳定的，当且仅当 $A$ 的所有特征值是实部小于等于 0。如果存在特征值的实部为 0，则该特征值不能出现在 Jordan block 中**
* **LTI 系统 $\dot{x} = Ax$ 是李雅普诺夫渐近稳定的，当且仅当 $A$ 的所有特征值是实部小于 0。**（线性系统渐近稳定，则大范围渐近稳定）

对于 LTI 系统，渐近稳定是要求最高的。如果渐近稳定，则必定内部稳定、BIBO稳定。

离散时间系统（自治系统）：

$$
\begin{cases}
x(k+1) = G(T)x(k) \\
y(k) = Cx(k)
\end{cases}
$$

* **LTI 离散系统是内部稳定的，当且仅当 $G$ 的所有特征值的幅值小于等于 1。如果存在特征值的幅值为 1，则该特征值不能出现在 Jordan block 中**
* **LTI 离散系统是李雅普诺夫渐近稳定的，当且仅当 $G$ 的所有特征值的幅值小于 1。**



## 李雅普诺夫稳定判据

### 连续时间系统

实际上，我们现在已经掌握了一种自治系统的稳定性判据，即特征值法（依据特征值来判定）。但很遗憾的是，这种方法的泛化性不强：在低阶的系统中，求取特征值是可行的，但对于高阶系统，求取特征值将会变得非常复杂与繁琐。

因此，有没有一种新的方法，能够普适的、简便的来判定自治系统的李雅普诺夫稳定性？李雅普诺夫本人，给出了**李雅普诺夫稳定判据（李雅普诺夫第二方法）**。（李雅普诺夫第一方法是求解系统微分方程，不具备普适性）

* **Lyapunov stability theorem**

  当系统状态在稳态点 $x_e$ 附近，如果存在一个李雅普诺夫函数 $V(x)$ 满足

  * $V(x)$ 是正定函数
  * $\dot{V}(x)$ 是负定函数

  则该系统是**渐近稳定**的。

  （这是一个充分条件，你可能找不到这样的一个李雅普诺夫函数，但不能说明该系统一定不稳定）

* **Lyapunov global asymptotic stability theorem**

  系统的稳态点 $x_e$ 位于原点，如果存在一个一阶微分连续的李雅普诺夫函数 $V(x, t)$ 满足

  * $V(x)$ 是正定函数
  * $\dot{V}(x)$ 是半负定函数
  * 在系统的任意运动轨迹上，除了原点，$\dot{V}(x)$ 不能恒等于0
  * 当 $\|x\| \rightarrow \infty$，$V(x, t) \rightarrow \infty$

  则系统的稳态点是**大范围渐近稳定**的 

  （这同样是一个充分条件，你可能找不到这样的一个李雅普诺夫函数，但不能说明该系统一定不稳定）

* **Lyapunov instability theorem**

  系统 $\dot{x}(t) = f(x(t), t)$ 的稳态点位于原点，如果存在一个李雅普诺夫函数 $V(x)$ 满足

  * $V(x)$ 是正定函数，在稳态点附近
  * $\dot{V}(x)$ 是证定函数，在同样的区域

  则该系统的稳态点是不稳定的



现在还剩哪些问题呢？

1. 上述的三个定理，都是充分条件，得看你找不找得到相应的李雅普诺夫函数
2. 没有给出一个李雅普诺夫函数的具体构造方法

因此，下面将给出充分必要条件以及相应的李雅普诺夫函数构造方法。



因为要求李雅普诺夫函数是正定的，因此一般的形式为正定二次型 $V(x) = x^TPx$ （当然可以不是二次型，这里也是考虑特殊形式）。

那么对应的，$\dot{V}(x) = x^T(A^TP + PA)x$，很显然如果 $Q = -(A^TP + PA)$ 是一个正定阵就好了。

* **necessary and sufficient**

  对于稳态点 $x_e$ 位于原点的 LTI 系统，系统是**渐近稳定**的，当且仅当，对于任意的**正定实对称阵** $Q$，李雅普诺夫等式
  
  $$
  A^TP+PA = -Q \quad \text{(usually let Q = I)}
  $$
  
  有**唯一**的**正定对称解** $P$。
  
  $$
  A^TP+PA = -I \\
  \Rightarrow P = \int_0^\infty e^{A^Tt}Qe^{At} dt
  $$
  
  该方法既能够判定一个系统是否是渐进稳定的，同时附带的给出了李雅普诺夫函数的构造方式。

### 离散时间系统

对于**离散时间系统**，同样存在相似的**充分必要**判据

$$
\begin{cases}
x(k+1) = G(T)x(k) \\
y(k) = Cx(k)
\end{cases}
$$

* **necessary and sufficient**

  对于稳态点 $x_e$ 位于原点的系统，系统是**渐近稳定**的，当且仅当，对于任意的**正定实对称阵** $Q$，离散李雅普诺夫等式
  
  $$
  G^TPG-G = -Q \quad \text{(usually let Q = I)}
  $$
  
  有**唯一**的**正定对称解** $P$。

### 非线性系统

对于**非线性系统**，仅存在相似的**充分**判据

$$
\dot{x} = f(x)
$$

雅可比矩阵：

$$
J(x) = \frac{\partial f(x)}{\partial x^T} = \begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \cdots & \frac{\partial f_2}{\partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial f_n}{\partial x_1} & \frac{\partial f_n}{\partial x_2} & \cdots & \frac{\partial f_n}{\partial x_n}
\end{bmatrix}
$$

* **sufficient theorem**

  非线性系统位于原点的稳态点是**渐近稳定**的，当对于任意的正定对称阵 $P$，
  
  $$
  Q = -[J^T(x)P + PJ(x)]
  $$
  
  是正定的。同时，相应的李雅普诺夫函数是
  
  $$
  V(x) = \dot{x}^TP\dot{x} = f^T(x)Pf(x)
  $$
  
  进一步，当 $\|x\| \rightarrow \infty$，$V(x) \rightarrow \infty$，则该系统是**大范围渐近稳定**的。





## 参数优化问题

* Problem

    考虑系统
    
    $$
    \dot{x} = A(\alpha)x, \quad x(0)=x_0
    $$
    
    选择合适的参数 $\alpha$，使得系统渐近稳定，同时最小化以下表达式：
    
    $$
    J = \int_0^\infty x^TQxdt
    $$
    
    其中 $Q$ 是已知的对称正定阵。

* Answer

  1. 使用充要条件来保证系统渐近稳定

     对于任意选取的对称正定阵 $R$，要求 $A^T(\alpha)P + PA(\alpha) = -R$ 有唯一的对称正定解 $P$，则
     
     $$
     V(x) = x^TPx, \quad \frac{dV(x)}{dt} = -x^TRx \\
     \int_0^\infty -x^TRxdt = \int_0^\infty \frac{dV(x)}{dt} dt = -V(0) = -x_0^TPx_0
     $$
     
     令 $R=Q$，则
     
     $$
     J = \int_0^\infty x^TQxdt = x_0^TPx_0
     $$

  2. 问题转换
     
     $$
     min \quad  J = x_0^TPx_0 \\
     s.t. \quad A^T(\alpha)P + PA(\alpha) = -Q
     $$
     







