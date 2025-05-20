---
layout:       post
title:        "【现代控制理论】- Controllability and Observability"
author:       "Orchid"
header-style: text
catalog:      true
hidden:       false
tags:
    - 控制
    - notes
---

在上一章节中我们已经讨论了对于状态空间模型的系统响应分析，对于以下的状态空间表达式：

$$
\begin{cases}
\dot{x} = Ax + Bu \\
y = Cx + Du
\end{cases}
$$

在已知系统初始状态 $x(0)$ 与系统的控制输入 $u(t)$，我们可以得到它的系统响应：

$$
x(t) = e^{At}x(0) + \int_0^t e^{A(t-\tau)} Bu(\tau) d\tau
$$

OK，我们已经学会了**建立系统、分析系统**，紧接着是不是应该尝试**控制系统、观测系统**？那么，紧接着我们需要回答以下两个问题：

1. **状态能控性问题**：在有限时间内，系统的控制输入能否使得系统从初始状态转移到指定要求的状态

   能控才能进一步实现最优控制，其揭示的是系统输入对于系统状态的控制能力

2. **状态能观性问题**：在有限时间内，能否通过系统的输出来估计系统在各时刻状态

   能观才能进一步实现反馈控制，其揭示的是系统输出对于系统状态的观测能力

<img src="https://notes.sjtu.edu.cn/uploads/upload_a359cb180187a14b1252a7b550c9006f.png" style="zoom:67%;" />

<img src="https://notes.sjtu.edu.cn/uploads/upload_cfd7099a09c948260b3c0813669d8261.png" style="zoom:67%;" />

## Controllability

* **Definition**

  考虑一个 n 维状态、p 维输入的系统：
  
  $$
  \dot{x}(t) = Ax(t) + Bu(t)
  $$
  
  System $(A, B)$ is said to be **controllable** if for any initial state $x(0) = x_0$ and any final state $x_1$, there exists a input that drives $x_0$ to $x_1$ in a finite time. 
  
  $$
  \forall x_0, x_1 \in \mathbb{R}^n , \ \exists t_1 > 0,  \ \exists u(t), \ t \in [0, t_1], \text{ such that }\\
  x_1 = e^{At_1} x_0 + \int_{0}^{t_1} e^{A(t_1-\tau)} B u(\tau) \mathrm{d}\tau
  $$

那么如何使得能控性定义中的等式成立呢？前人提出了一个非常巧妙的想法：

$$
\text{let } u(t) = -B^T e^{A^T(t_1 - t)} W_c^{-1}(t_1) \left( e^{At_1} x_0 - x_1 \right), \\
\text{where } W_c(t) = \int_{0}^{t} e^{A\tau} BB^T e^{A^T \tau} \mathrm{d}\tau  = \int_{0}^{t} e^{A(t-\tau)} BB^T e^{A^T (t-\tau)} \mathrm{d}\tau.
$$

这样是不是就实现了，系统的能控性呢？注意，**需要考虑 $W_c(t)$ 的可逆性**！（$W_c(t)$ 为**能控格拉姆矩阵**）



* **定理 1：系统 $(A,B)$ 是能控的，当且仅当 $W_c(t) \in R^{n \times n}$ 在任意时刻 $t>0$ 是可逆的（非奇异的）**

  **充分性：**
  
  $$
  \begin{align*}
  x(t_1) &= e^{At_1} x_0 + \int_{0}^{t_1} e^{A(t_1 - \tau)} Bu(\tau) \mathrm{d}\tau \\
  &= e^{At_1} x_0 - \int_{0}^{t_1} e^{A(t_1 - \tau)} BB^T e^{A^T (t_1 - \tau)} W_c^{-1} (t_1) \left( e^{At_1} x_0 - x_1 \right) \mathrm{d}\tau \\
  &= e^{At_1} x_0 - \left( \int_{0}^{t_1} e^{A(t_1 - \tau)} BB^T e^{A^T (t_1 - \tau)} \mathrm{d}\tau \right) W_c^{-1} (t_1) \left( e^{At_1} x_0 - x_1 \right) \\
  &= e^{At_1} x_0 - e^{At_1} x_0 + x_1 = x_1
  \end{align*}
  $$
  
  **必要性：**

  （假设系统 $(A，B)$ 是能控的，但是 $W_c(t_1)$ 是奇异的）

  因为 $W_c(t_1)$ 是奇异的，因此存在**非零向量** $v$ 使得：
  
  $$
  v^T W_c(t_1) v = \int_{0}^{t_1} v^T e^{A(t_1 - \tau)} BB^T e^{A^T (t_1 - \tau)} \mathrm{d}\tau = \int_{0}^{t_1} ||B^T e^{A^T (t_1 - \tau)} v||^2 \mathrm{d}\tau = 0
  $$
  
  进一步可以得到，$B^T e^{A^T (t_1 - \tau)} v \equiv 0$ 或是 $v^T e^{A (t_1 - \tau)} B \equiv 0, \forall \tau \in [0, t_1]$

  因为系统 $(A,B)$ 是能控的，设系统的初始状态为 $x_0 = e^{-At_1}v$，系统的目标状态为 $x(t_1) = 0$
  
  $$
  x(t_1) = e^{At_1} x_0 + \int_{0}^{t_1} e^{A(t_1 - \tau)} Bu(\tau) \mathrm{d}\tau = v + \int_{0}^{t_1} e^{A(t_1 - \tau)} Bu(\tau) \mathrm{d}\tau = 0
  $$
  
  两边左乘 $v^T$，可以得到：
  
  $$
  0 = v^T v + \int_{0}^{t_1} v^T e^{A(t_1 - \tau)} Bu(\tau) \mathrm{d}\tau = ||v||^2 + 0
  $$
  
  这与 $v$ 为非零向量矛盾！



* **定理 2：系统 $(A,B)$ 是能控的，当且仅当 $G_c = [B \ AB \ A^2B \ \cdots \ A^{n-1}B] \in R^{n \times np}$ 是行满秩的**

  **充分性：如果不可控，则不满秩**

  由定理 1，若系统不可控，则 $W_c(t)$ 是奇异的。因此存在非零向量 $v$ 使得：
  
  $$
  ve^{At} B = vB + vtAB + v \frac{t^2}{2!} A^2 B + \cdots = 0
  $$
  
  考虑 $t=0$，有 $vB = 0$

  等式两边求导，再考虑 $t=0$，有 $vAB = vA^2B = \cdots = vA^{n-1}B = 0$

  进而得到 $v [B \ AB \ A^2B \ \cdots \ A^{n-1}B] = 0$，故 $G_c$ 非行满秩！

  **必要性：如果不满秩，则不可控**

  因为 $G_c$ 不满秩，则存在非零向量 $v$ 使得 $vG_c$ = 0 或 $vB = vAB = vA^2B = \cdots = vA^{n-1}B = 0$ 

  由上一章提到的 **Cayley–Hamilton Theorem method**：
  
  $$
  e^{At} = \alpha_0(t)I + \alpha_1(t)A + \alpha_2(t)A^2 + \cdots + \alpha_{n-1}(t)A^{n-1}
  $$
  
  所以 $e^{At}B = [B \ AB \ A^2B \ \cdots \ A^{n-1}B] [\alpha_0(t) \ \alpha_1(t) \ \cdots \ \alpha_{n-1}(t)]^T$
  
  进而 $ve^{At}B = 0$，则 $W_c(t)$ 是奇异的。故系统 $(A,B)$ 是不可控的！



* **定理 3：系统 $(A,B)$ 是能控的，当且仅当对于 $A$ 的任意特征值 $\lambda$， $rank[\lambda I - A \ B] = n$**

  **必要性：如果不满秩，则不可控**

  因为 $rank[\lambda I - A \ B] < n$，则存在非零向量 $v$ 使得 $v[\lambda I - A \ B] = 0$，即 $vA = \lambda v, \ vB = 0$

  所以有 $vAB = \lambda vB = 0, \cdots, vA^{n-1}B = \lambda^{n-1}vB = 0$，即 $v[B \ AB \ A^2B \ \cdots \ A^{n-1}B] = 0$

  由定理 2，可知系统是不可控的！

  **充分性：如果不可控，则不满秩**

  （省略，好像不好证）



上述的三个定理，在说明系统可控的同时，都是为了使得控制器 $u(t) = -B^T e^{A^T(t_1 - t)} W_c^{-1}(t_1) \left( e^{At_1} x_0 - x_1 \right)$ 成立。但是这是一个**开环控制器**，且涉及矩阵积分，是一个对参数敏感且复杂的控制器，在实际应用中几乎不会使用。

一个更有效的方法是，采用简单的**线性反馈控制器** $u = r - Kx$，使得系统状态渐进地趋于目标状态。





## Observability

首先，能观性问题考虑能否通过系统的输出来估计系统在**各时刻状态**，实际上**只用估计系统初始时刻的状态**，就能根据系统的控制输入推断各时刻状态。在系统能否观测这个问题中，系统的输出 $y(t)$ 是可以测量的，系统的控制输入 $u(t)$ 也是已知的。因此，考虑输出方程：

$$
y(t) = Cx(t) + Du(t) = Ce^{At} x_0 + C \int_{0}^{t} e^{A(t-\tau)} Bu(\tau) \mathrm{d}\tau + Du(t)
$$

进而，可以定义：

$$
\bar{y}(t) = y(t) - C \int_{0}^{t} e^{A(t-\tau)} Bu(\tau) \mathrm{d}\tau - Du(t) = Ce^{At} x_0
$$

甚至可以**直接将控制输入 $u \equiv 0$**（**系统的输入不改变系统的能观测性**），则直接有：

$$
\bar{y}(t) = y(t)= Ce^{At} x_0
$$

因此，能否从系统输出 $\bar{y} \text{ or } y$ 中观测得到系统的初始状态量，取决于 $A \text{ and } C$。

* **Definition：**

  The system is said to be observable if for any unknown initial state $x(0)$, there exists a finite $t_1 > 0$ such that $x(0)$ can be exactly determined over $[0, t_1]$ from $u$ and $y$.

  （当然，实际中会将 $u \equiv 0$，则只考虑 $y$）



那么，如何使得系统可观测呢？参考能控性的巧妙想法（构造一个可逆方阵），这里同样可以提供一个非常巧妙的 idea：

$$
\text{let } W_o(t) =  \int_{0}^{t} e^{A^T\tau} C^TC e^{A\tau} \mathrm{d}\tau \\
$$

* **定理 1：系统 $(A,C)$ 是可观的，当且仅当矩阵 $W_o(t) \in R^{n \times n}$ 在任意时刻 $t>0$ 是非奇异的**

  **充分性：**如果是非奇异的，则客观
  
  $$
  \begin{align}
  &y(t) = Ce^{At} x_0 \\
  \Rightarrow &e^{A^Tt} C^T y(t) = e^{A^Tt} C^TCe^{At} x_0 \\
  \Rightarrow &\int_{0}^{t_1} e^{A^Tt} C^T y(t)\mathrm{d}t = \int_{0}^{t_1} e^{A^Tt} C^TCe^{At} x_0 \mathrm{d}t = W_o(t_1) x_0 \\
  &\text{If } W_o(t_1) \text{ is not singular, then} \\
  \Rightarrow & x_0 = W_o^{-1}(t_1) \int_{0}^{t_1} e^{A^Tt} C^T y(t)\mathrm{d}t
  
  \end{align}
  $$
  
  **必要性：**如果是奇异的，则不客观

  因为 $W_o(t)$ 是奇异的，所以存在非零向量 $v$，使得 $Ce^{At}v = 0 \ \text{for all} \ t$

  则 $y(t) = Ce^{At}x_0 = Ce^{At}(x_0 + kv)$

  系统输出不能唯一确定系统的初始状态量，故系统不可观！



同样，可以仿照能控性的定理，继续给出**能观性的等价定理**：

* **定理 2：系统 $(A,C)$ 是可观的，当且仅当矩阵 $G_o = \begin{bmatrix} C \\ CA \\ \vdots \\ CA^{n-1} \end{bmatrix}$ 是列满秩的**
* **定理 3：系统 $(A,C)$ 是可观的，当且仅当对于 $A$ 的任意特征值 $\lambda$， $rank \begin{bmatrix} \lambda I -A \\ C \end{bmatrix} = n$**

同样的，上述三个定理在说明了系统能观性的同时，也给出了一种估计系统初始状态量的方法 $x_0 = W_o^{-1}(t_1) \int_{0}^{t_1} e^{A^Tt} C^T y(t)\mathrm{d}t$，同样的这是一种**开环估计器**，即根据已有信息一次性估计出来的，同样因为涉及积分，是一种参数敏感且复杂的状态估计器。



## 结构分解

根据上述两个小节，我们已经学会了如何判断一个系统是否是完全可观，或者是否是完全能控的。但是，我们往往会遇到**部分能控**或是**部分可观**的系统，在这些系统中，并不是所有的状态量都是能控或可观的。因此，可以通过线性变换的方式，对状态空间进行变换，使得能控或可观的状态量和不能控、不可观的状态量分解开来，那么将上述这种变换或分解为**标准分解**。

### 能控性分解

已知系统状态空间模型：

$$
\begin{cases}
\dot{x} = Ax + Bu \\
y = Cx + Du
\end{cases}
$$

假设能控性矩阵 $G_c$ 的秩 $n_1 < n$，即系统并非完全能控。

* **定理 1：** 存在非奇异的变换矩阵 $P$ ，对系统的状态空间表达式进行线性变换 $x = P\widetilde{x} $，
  
  $$
  \begin{cases}
  \dot{\widetilde{x}} = \widetilde{A}\,\widetilde{x} + \widetilde{B}\,u \\
  y = \widetilde{C}\,\widetilde{x}
  \end{cases}
  $$
  
  其中
  
  $$
  \widetilde{A} = P^{-1} A P = \begin{bmatrix} \widetilde{A}_{11} & \widetilde{A}_{12} \\ \hdashline 0 & \widetilde{A}_{22} \end{bmatrix},\quad\widetilde{B} = P^{-1} B = \begin{bmatrix} \widetilde{B}_1 \\ \hdashline 0 \end{bmatrix},\quad\widetilde{C} = C P = \begin{bmatrix} \widetilde{C}_1 & \widetilde{C}_2 \end{bmatrix}
  $$
  
  此时，系统的前 $n_1$ 维构成了能控的子系统，后 $n-n_1$ 维为不能控系统：
  
  $$
  \dot{\tilde{x}}_1 = \widetilde{A}_{11}\,\tilde{x}_1 + \widetilde{A}_{12}\,\tilde{x}_2 + \widetilde{B}_1 u \\
  \dot{\tilde{x}}_2 = \widetilde{A}_{22}\,\tilde{x}_2
  $$

  <img src="https://notes.sjtu.edu.cn/uploads/upload_28d7a7420acf2cda0106b92a3296d4ea.png" style="zoom:67%;" />
  
* **变换矩阵 $P$ 的构造方式：**
  1. 在能控性矩阵 $G_c = [B \ AB \ A^2B \ \cdots \ A^{n-1}B]$ 中选择 $n_1$ 个线性无关的列向量
  2. 将所得的列向量作为矩阵 $P$ 的前 $n_1$ 列，其余列可以在保证 $P$ 可逆的前提下任意选择

* **定理 2：** 能控子系统的传递函数与原系统的传递函数一致。
  
  $$
  \begin{align}
  G(s) &= C(sI - A)^{-1}B = \widetilde{C}(sI - \widetilde{A})^{-1}\widetilde{B} \\
  
  &= \begin{bmatrix} \widetilde{C}_1 & \widetilde{C}_2 \end{bmatrix} \begin{bmatrix} sI - \widetilde{A}_{11} & -\widetilde{A}_{12} \\ 0 & sI - \widetilde{A}_{22} \end{bmatrix}^{-1} \begin{bmatrix} \widetilde{B}_1 \\ 0 \end{bmatrix} \\
  
  &= \widetilde{C}_1 [sI - \widetilde{A}_{11}]^{-1} \widetilde{B}_1 = \widetilde{G}_1(s)
  \end{align}
  $$
  

### 能观性分解

已知系统状态空间模型：

$$
\begin{cases}
\dot{x} = Ax + Bu \\
y = Cx + Du
\end{cases}
$$

假设能观性矩阵 $G_o$ 的秩 $n_1 < n$，即系统并非完全能观。

* **定理 1：** 存在非奇异的变换矩阵 $P$ ，对系统的状态空间表达式进行线性变换 $x = P\widetilde{x} $，
  
  $$
  \begin{cases}
  \dot{\widetilde{x}} = \widetilde{A}\,\widetilde{x} + \widetilde{B}\,u \\
  y = \widetilde{C}\,\widetilde{x}
  \end{cases}
  $$
  
  其中
  
  $$
  \widetilde{A} = P^{-1} A P = \begin{bmatrix} \widetilde{A}_{11} & 0 \\ \hdashline \widetilde{A}_{21} & \widetilde{A}_{22} \end{bmatrix},\quad\widetilde{B} = P^{-1} B = \begin{bmatrix} \widetilde{B}_1 \\ \hdashline \widetilde{B}_2 \end{bmatrix},\quad\widetilde{C} = C P = \begin{bmatrix} \widetilde{C}_1 & 0 \end{bmatrix}
  $$
  
  此时，系统的前 $n_1$ 维构成了能观的子系统，后 $n-n_1$ 维为不能观系统：
  
  $$
  y = \widetilde{C}_1\widetilde{x}_1
  $$
  
  <img src="https://notes.sjtu.edu.cn/uploads/upload_3d7efe3b41c074bf860e830c2f8fafda.png" style="zoom:67%;" />
  
* **变换矩阵 $P$ 的构造方式：**

  1. 在能控性矩阵  $G_o = \begin{bmatrix} C \\ CA \\ \vdots \\ CA^{n-1} \end{bmatrix}$ 中选择 $n_1$ 个线性无关的行向量
  2. 将所得的行向量作为矩阵 $P^{-1}$ 的前 $n_1$ 行，其余行可以在保证 $P^{-1}$ 可逆的前提下任意选择

* **定理 2：** 能观子系统的传递函数与原系统的传递函数一致。
  
  $$
  \begin{align}
  G(s) &= C(sI - A)^{-1}B = \widetilde{C}(sI - \widetilde{A})^{-1}\widetilde{B} \\
  
  &= \begin{bmatrix} \widetilde{C}_1 & 0 \end{bmatrix} \begin{bmatrix} sI - \widetilde{A}_{11} & 0 \\ -\widetilde{A}_{21} & sI - \widetilde{A}_{22} \end{bmatrix}^{-1} \begin{bmatrix} \widetilde{B}_1 \\ \widetilde{B}_2 \end{bmatrix} \\
  
  &= \widetilde{C}_1 [sI - \widetilde{A}_{11}]^{-1} \widetilde{B}_1 = \widetilde{G}_1(s)
  \end{align}
  $$
  

### 卡尔曼分解

能控性分解是将系统分为能控和不能控的部分，能观性分解是将系统分解为能观和不能观的部分，而卡尔曼分解则是将两者结合在一起，将系统划分为能控且能观、能控但不能观、不能控但能观、不能控且不能观四个部分。

* **定理：** 对于既不完全能控又不完全能观的线性系统状态空间表达式
  
  $$
  \begin{cases}
  \dot{x} = Ax + Bu \\
  y = Cx + Du
  \end{cases}
  $$
  
  经过线性变换，可以化为下列形式：
  
  $$
  \begin{bmatrix}
  \dot{\tilde{x}}_1 \\
  \dot{\tilde{x}}_2 \\
  \dot{\tilde{x}}_3 \\
  \dot{\tilde{x}}_4
  \end{bmatrix}
  =
  \begin{bmatrix}
  \widetilde{A}_{11} & 0 & \widetilde{A}_{13} & 0 \\
  \widetilde{A}_{12} & \widetilde{A}_{22} & \widetilde{A}_{23} & \widetilde{A}_{24} \\
  0 & 0 & \widetilde{A}_{33} & 0 \\
  0 & 0 & \widetilde{A}_{43} & \widetilde{A}_{44}
  \end{bmatrix}
  \begin{bmatrix}
  \tilde{x}_1 \\
  \tilde{x}_2 \\
  \tilde{x}_3 \\
  \tilde{x}_4
  \end{bmatrix}
  +
  \begin{bmatrix}
  \widetilde{B}_1 \\
  \widetilde{B}_2 \\
  0 \\
  0
  \end{bmatrix}
  u
  $$

  $$
  y = \begin{bmatrix} \widetilde{C}_1 & 0 & \widetilde{C}_3 & 0 \end{bmatrix} \begin{bmatrix} \tilde{x}_1 \\ \tilde{x}_2 \\ \tilde{x}_3 \\ \tilde{x}_4 \end{bmatrix}
  $$

  其中，$\tilde{x}_1$ 是既能控又能观的，$\tilde{x}_2$ 是能控但不能观的，$\tilde{x}_3$ 是不能控但能观的，$\tilde{x}_4$ 是既不能控又不能观的。



## Minimal Realization

假设 $G(s)$ 是一个 proper rational transfer matrix。我们在第一章的时候就学习过，状态空间表达式和传递函数之间的关系：

$$
G(s) = C(sI-A)^{-1}B+D
$$

我们称 $(A,B,C,D)$ 是 $G(s)$ 的一种**实现**。很显然这种实现不是唯一的（我可以对状态方程进行线性变换，或给传递函数上下同乘某一项等等）。

我们称在所有实现中，唯独最低的 $(A,B,C,D)$ 状态空间表达式，为 $G(s)$ 的**最小实现**。（当然，最小实现也不是唯一的）



* **定理：** $(A,B,C,D)$ 是最小实现，当且仅当系统 $(A,B)$ 能控，系统 $(A,C)$ 能观。



对于 proper rational 的传递函数 $G(s) = \frac{N(s)}{D(s)} = \frac{\beta_1 s^3 + \beta_2 s^2 + \beta_3 s + \beta_4}{s^4 + \alpha_1 s^3 + \alpha_2 s^2 + \alpha_3 s + \alpha_4}$，在所有实现中，有两种特殊的实现：能控标准型、能观标准型

* **能控标准型**
  
  $$
  \dot{x} = \begin{bmatrix}
  -\alpha_1 & -\alpha_2 & -\alpha_3 & -\alpha_4 \\
  1 & 0 & 0 & 0 \\
  0 & 1 & 0 & 0 \\
  0 & 0 & 1 & 0
  \end{bmatrix} x + \begin{bmatrix} 1 \\ 0 \\ 0 \\ 0 \end{bmatrix} u \\
  y = \begin{bmatrix} \beta_1 & \beta_2 & \beta_3 & \beta_4 \end{bmatrix} x + d u
  $$

  * 任何一个传递函数 $G(s)$ 都能被实现成一个能控系统（但不一定能观）
  * **能控系统能观，当且仅当传递函数的分子与分母互质（没有可以约分的项）**（此时这是一种最小实现）

* **能观标准型**
  
  $$
  \dot{x} = \begin{bmatrix}
  -\alpha_1 & 1 & 0 & 0 \\
  -\alpha_2 & 0 & 1 & 0 \\
  -\alpha_3 & 0 & 0 & 1 \\
  -\alpha_4 & 0 & 0 & 0
  \end{bmatrix} x + \begin{bmatrix} \beta_1 \\ \beta_2 \\ \beta_3 \\ \beta_4 \end{bmatrix} u \\
  
  y = \begin{bmatrix} 1 & 0 & 0 & 0 \end{bmatrix} x + d u
  $$

  * 任何一个传递函数 $G(s)$ 都能被实现成一个能观系统（但不一定能控）
  * **能观系统能控，当且仅当传递函数的分子与分母互质（没有可以约分的项）**（此时这是一种最小实现）



* **定理：** 对于 SISO 系统，其状态空间表达式 $(A,b,c)$ 是能控且能观的（最小实现），当且仅当其转化成的**传递函数不会出现零极点相消**（转化成的传递函数分子和分母互质）

  
