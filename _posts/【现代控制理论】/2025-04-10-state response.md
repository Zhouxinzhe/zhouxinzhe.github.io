---
layout:       post
title:        "【现代控制理论】- State Response"
author:       "Orchid"
header-style: text
catalog:      true
hidden:       false
tags:
    - 控制
    - notes
---

## Response Analysis

在上一章中，我们已经建立了状态空间模型，并对其进行了一定的讨论与分析。

$$
\begin{cases}
\dot{x} = Ax + Bu \\
y = Cx + Du
\end{cases}
$$

新的问题是：如果知道系统的初始状态和控制输入，能够求解每一时刻的系统状态和输出吗？这一小节将给出系统的 response analysis。

根据状态空间表达式，系统的响应（状态变化）可以看作初始状态 $x(0) = x_0$ 和控制输入 $u(t)$ 的共同作用下的结果。而对于**线性系统**，一定满足叠加定理。因此系统的响应可以看作初始状态 $x(0) = x_0$ 和控制输入 $u(t)$ 两者单独作用后的叠加。

<img src="https://notes.sjtu.edu.cn/uploads/upload_cff035fff5659f496377b85463c5c598.png" style="zoom:67%;" />

### 零输入响应

假设系统的控制输入 $u(t) = 0$，系统状态量的变化完全取决于系统的初始状态量，即：

$$
\dot{x}(t) = Ax(t), t_0 = 0
$$

如果将上述式子看成一个微分方程，那么很容易得到解：

$$
x(t) = e^{At}x(0)
$$

但通常意义上的微分方程求解的是一维的问题，当 $A$ 是一个 $n\times n$ 的矩阵时，上述解是否正确呢？只需回代回状态方程验证即可：

$$
\begin{align}
\dot{x}(t) &= \frac{d}{dt}e^{At}x(0) \\
		   &= \frac{d}{dt} \left( I + At + \frac{1}{2!} A^2 t^2 + \cdots + \frac{1}{n!} A^n t^n + \cdots \right)x(0) \\
		   &= (A + \frac{1}{2!} A^2 (2) t + \cdots + \frac{1}{n!} A^n (n) t^{n-1} + \cdots)x(0) \\
		   &= A \left( I + At + \cdots + \frac{1}{(n-1)!} A^{n-1} t^{n-1} + \cdots \right)x(0) \\
		   &= A e^{At}x(0)\quad(t = 0, \quad e^{At} = I)
\end{align}
$$

所以，**求解系统的零输入响应的关键，就在于计算 $e^{At}$**。下面将给出几种计算 $e^{At}$ 的方法。

1. **直接法**

   直接使用泰勒展开式求解：
   
   $$
   e^{At} =  I + At + \frac{1}{2!} A^2 t^2 + \cdots + \frac{1}{n!} A^n t^n + \cdots 
   $$
   
   但是需要计算 $A^n$，往往计算比较复杂

2. **线性变换法**（化为标准型）

   * 标准对角阵

   由上一章的线性变换小节可知，可以通过线性变换的方式将一些状态矩阵 $A$ 变换为对角阵 $\bar{A} = P^{-1}AP$，而对角阵的 n 阶次 $\bar{A}^n$ 是容易计算的。
   
   $$
   P^{-1}AP = \bar{A} = \begin{bmatrix}
   \lambda_1 & 0 & \cdots & 0 \\
   0 & \lambda_2 & \cdots & 0 \\
   \vdots & \vdots & \ddots & \vdots \\
   0 & 0 & \cdots & \lambda_n
   \end{bmatrix} \\
   \quad A = P\bar{A}P^{-1} \\
   A^n = (P\bar{A}P^{-1})^n = P\bar{A}^nP^{-1}
   $$
   
   因此，$e^{At}$ 的计算可以转化为：
   
   $$
   \begin{align}
   e^{At} &= e^{P\bar{A}P^{-1} t} \\
   &= I + (P\bar{A}P^{-1})t + \frac{1}{2!}(P\bar{A}P^{-1})^2 t^2 + \cdots + \frac{1}{n!}(P\bar{A}P^{-1})^n t^n + \cdots \\
   &= P^{-1}IP + P\bar{A}P^{-1} t + \frac{1}{2!}P\bar{A}^2P^{-1} t^2 + \cdots + \frac{1}{n!}P\bar{A}^nP^{-1} t^n + \cdots \\
   &= P \left( I + \bar{A}t + \frac{1}{2!} \bar{A}^2 t^2 + \cdots + \frac{1}{n!} \bar{A}^n t^n + \cdots \right) P^{-1} \\
   &= P e^{\bar{A}t} P^{-1} \\
   &= P \begin{bmatrix}
   e^{\lambda_1t} & 0 & \cdots & 0 \\
   0 & e^{\lambda_2t} & \cdots & 0 \\
   \vdots & \vdots & \ddots & \vdots \\
   0 & 0 & \cdots & e^{\lambda_nt}
   \end{bmatrix} P^{-1}
   \end{align}
   $$

   * Jordan 标准型

   由上一章的线性变换小节可知，可以通过线性变换的方式将一些状态矩阵 $A$ 变换为若当阵 $\bar{A} = P^{-1}AP$，而若当阵的 n 阶次 $\bar{A}^n$ 虽然没有对角阵容易计算，但相对于普通矩阵 $A$ 仍算有规律可循。

   下面给出一个一般的例子：
   
   $$
   P^{-1}AP = \bar{A} = \begin{bmatrix}
   \lambda_1 & 1 & &  &  &  \\
    & \lambda_1 & 1 &  &  &  \\
    &  & \lambda_1 &  &  & \\
    &  &  & \lambda_2 & 1 &  \\
    &  &  &  & \lambda_2 & \\
    &  &  &  &  & \lambda_3
   \end{bmatrix}
   $$
   
   其矩阵指数 $e^{At}$ 应有以下形式：
   
   $$
   e^{At} = P
   \begin{bmatrix}
   e^{\lambda_1 t} & t e^{\lambda_1 t} & \frac{1}{2} t^2 e^{\lambda_1 t} & 0 & 0 & 0 \\
   0 & e^{\lambda_1 t} & t e^{\lambda_1 t} & 0 & 0 & 0 \\
   0 & 0 & e^{\lambda_1 t} & 0 & 0 & 0 \\
   \hdashline
   0 & 0 & 0 & e^{\lambda_2 t} & t e^{\lambda_2 t} & 0 \\
   0 & 0 & 0 & 0 & e^{\lambda_2 t} & 0 \\
   \hdashline
   0 & 0 & 0 & 0 & 0 & e^{\lambda_3 t}
   \end{bmatrix}
   P^{-1}
   $$
   
   证明从略（较复杂）。

3. **拉式变换法**
   
   $$
   \begin{align}
   \dot{x} &= Ax \\
   \Rightarrow sX(s) - x(0) &= AX(s) \\
   \Rightarrow X(s) &= (sI-A)^{-1}x(0) \\
   \Rightarrow x(t) &= L^{-1}[(sI-A)^{-1}]x(0) \\
   \Rightarrow e^{At} &= L^{-1}[(sI-A)^{-1}]
   \end{align}
   $$
   
   所以**求解矩阵指数 $e^{At}$ 的关键转化为了 $L^{-1}[(sI-A)^{-1}]$**

4. **Cayley–Hamilton Theorem method**

   首先考虑矩阵 $A$ 的特征多项式：
   
   $$
   p(\lambda) = det(\lambda I - A) = \lambda^n + a_{n-1}\lambda^{n-1} + \cdots + a_0
   $$
   
   将上述特征多项式中的 $\lambda$ 代替为矩阵 $A$ 很容易得到：
   
   $$
   p(A) = det(A - A) = A^n + a_{n-1}A^{n-1} + \cdots + a_0I = 0
   $$
   
   因此，可以得到：
   
   $$
   A^n = -a_0 I - a_1 A - \cdots - a_{n-1} A^{n-1} \\
   A^{n+1} = -a_0 A - a_1 A^2 - \cdots - a_{n-1} A^n \\
   \vdots
   $$
   
   所以，指数矩阵 $e^{At}$ 理论上可以转化为：
   
   $$
   e^{At} = \alpha_0(t)I + \alpha_1(t)A + \alpha_2(t)A^2 + \cdots + \alpha_{n-1}(t)A^{n-1}
   $$
   
   因此关键在于求解 $\alpha_0(t), \alpha_1(t), \cdots, \alpha_{n-1}(t)$。我们着眼于 $A$ 的特征值，对于特征值我们有：
   
   $$
   \lambda_i^n + a_{n-1}\lambda_i^{n-1} + \cdots + a_0 = 0
   $$
   
   因此，$e^{\lambda_i t}$ 可以表示为：
   
   $$
   e^{\lambda_1 t} = \alpha_0(t) + \alpha_1(t)\lambda_1 + \cdots + \alpha_{n-1}(t)\lambda_1^{n-1} \\
   e^{\lambda_2 t} = \alpha_0(t) + \alpha_1(t)\lambda_2 + \cdots + \alpha_{n-1}(t)\lambda_2^{n-1} \\
   \vdots \\
   e^{\lambda_n t} = \alpha_0(t) + \alpha_1(t)\lambda_n + \cdots + \alpha_{n-1}(t)\lambda_n^{n-1}
   $$
   
   因此，我们可以通过以下等式求解 $\alpha_0(t), \alpha_1(t), \cdots, \alpha_{n-1}(t)$：
   
   $$
   \begin{bmatrix}
   1 & \lambda_1 & \lambda_1^2 & \cdots & \lambda_1^{n-1} \\
   1 & \lambda_2 & \lambda_2^2 & \cdots & \lambda_2^{n-1} \\
   \vdots & \vdots & \ddots & \vdots \\
   1 & \lambda_n & \lambda_n^2 & \cdots & \lambda_n^{n-1}
   \end{bmatrix}
   \begin{bmatrix}
   \alpha_0(t) \\
   \alpha_1(t) \\
   \vdots \\
   \alpha_{n-1}(t)
   \end{bmatrix}
   =
   \begin{bmatrix}
   e^{\lambda_1 t} \\
   e^{\lambda_2 t} \\
   \vdots \\
   e^{\lambda_n t}
   \end{bmatrix}
   $$
   
   该方法比较复杂，不推荐使用。上述推导也十分粗略，存在很多逻辑漏洞，但大概是这么回事。

### 零状态响应

假设系统的初始状态 $x(t_0) = 0$，则系统状态的变化完全取决于系统的控制输入：

$$
\dot{x} = Ax + Bu \\
\dot{x} - Ax = Bu \\
e^{-At} (\dot{x} - Ax) = e^{-At}Bu \\
\frac{d}{dt}(e^{-At}x) = e^{-At}Bu \\
e^{-At} x(t) = \int_0^t e^{-A\tau} Bu(\tau) d\tau
$$

因此，可以得到零状态响应：

$$
x(t) = \int_0^t e^{A(t-\tau)} Bu(\tau) d\tau
$$

当然也可以从拉氏变换的角度推得上述结论。

### 系统响应

OK，现在已经分别推导了系统的零输入响应和零状态响应，则**系统的响应 = 零输入响应 + 零状态响应**：

$$
x(t) = e^{At}x(0) + \int_0^t e^{A(t-\tau)} Bu(\tau) d\tau
$$

可以看到在系统响应表达式中，有一个非常重要的项，$e^{At}$。围绕着一项，可以对系统响应做新的解释：系统的输入包括系统初始时刻的状态 $x(0)$ 以及连续输入的每一时刻的 $u(t)$，而 $e^{At}$ 的作用就是将输入作用到 $x(t)$；比如 $x(0)$ 的作用时间是 $t$ ，所以是 $e^{At}x(0)$，而 $\tau$ 时刻输入的 $u(\tau)$ 其作用时间为 $t-\tau$，所以是 $e^{A(t-\tau)} Bu(\tau)$。将所有输入对最后状态的作用求和（积分），即可求得最终状态 $x(t)$。

因此，**$e^{At}$ 的作用就是将某一时刻的系统输入转移到最终的系统状态上**，我们将其命名为**状态转移矩阵**（State Transition Matrix）。我们将在下一节对这一重要概念进行进一步讨论。



## State Transition Matrix

在上一节中，我们已经简单阐述了 $e^{At}$ 的作用，即状态转移，所以称其为**状态转移矩阵**，这里用一个**新的符号表示 $\Phi(t, t_0)$** （在线性定常系统中用 **$\Phi(t-t_0)$** 表示）。它不仅是时间 $t$ 的函数，也是初始时刻 $t_0$ 的函数，直观上这么定义也是很有道理的。那么系统响应可以用状态转移矩阵重新表示（暂时讨论线性定常系统）：

$$
x(t) = \Phi(t-t_0)x(0) + \int_0^t \Phi(t-\tau) Bu(\tau) d\tau
$$

从状态转移矩阵的定义 $\Phi(t) = e^{At}$，可以发现其许多**性质**：

* $\Phi(0) = I, \quad \dot{\Phi}(t) = A \Phi(t)$
* $\forall t, s \quad \Phi(t+s) = \Phi(t) \Phi(s) $
* $\Phi^{-1}(t) = \Phi(-t)$



> *接下来，非常奇怪，引入了一个叫**基础解阵**的东西，没搞清楚为什么？*

因为，$\dot{x} = Ax$ 有且仅有 $n$ 个线性无关的解，这 $n$ 个解可以构成**基础解阵** $\Psi(t) \in \mathbb{R}^{n \times n}$，显然其满足：

$$
\dot{\Psi}(t) = A \Psi(t), \quad \Psi(t_0) = H, \quad t \geq t_0
$$

其中 $H$ 是非奇异的常数矩阵，与具体方程的解有关。

**状态转移矩阵可以被基础解阵表示：**

$$
\Phi(t-t_0) = \Psi(t)\Psi^{-1}(t_0), \quad t \geq t_0
$$

证明：

$$
\text{Property:}\quad\dot{\Phi}(t - t_0) = \dot{\Psi}(t) \Psi^{-1}(t_0) = A \Psi(t) \Psi^{-1}(t_0) = A \Phi(t - t_0) \\

\text{Initial condition:}\quad \Phi(0) = \Phi(t_0 - t_0) = \Psi(t_0) \Psi^{-1}(t_0) = I
$$

基于基础解阵的表示，状态转移矩阵又可以发现新的**性质：**

- $\Phi(0) = \Psi(t_0) \Psi^{-1}(t_0) = I$
- $\Phi^{-1}(t - t_0) = \Psi(t_0) \Psi^{-1}(t) = \Phi(t_0 - t)$
- $\Phi(t_2 - t_0) = \Psi(t_2) \Psi^{-1}(t_0) = \Psi(t_2) \Psi^{-1}(t_1) \Psi(t_1) \Psi^{-1}(t_0) = \Phi(t_2 - t_1) \Phi(t_1 - t_0)$
- $\Phi(t_2 + t_1) = \Phi(t_2 - (-t_1)) = \Phi(t_2 - 0) \Phi(0 - (-t_1)) = \Phi(t_2) \Phi(t_1)$
- $\Phi(m \cdot t) = \Phi(t + t + \cdots + t) = \Phi(t) \Phi(t) \cdots \Phi(t) = [\Phi(t)]^m.$

**注意，状态转移矩阵只由 $A$ 决定，基础解阵只是一种表示方式**



## Time-variant dynamical system

上述的讨论都是基于线性定常系统的，但实际中，状态表达式的系数 $A(t)$ 可能是时变的。考虑以下时变系统：

$$
\dot{x}(t) = A(t) x(t), \quad x(t_0) = x_0, \quad x(t) \in \mathbb{R}^n, \quad t \geq t_0
$$

如何求解其系统状态呢？第一反应是和定常系统一样求解：

$$
\frac{\mathrm{d}x(t)}{\mathrm{d}t} = A(t)x(t) \Rightarrow \frac{1}{x(t)} \mathrm{d}x(t) = A(t) \mathrm{d}t \Rightarrow x(t) = e^{\int_{t_0}^{t} A(\tau) \mathrm{d}\tau} x_0
$$

**但这样正确吗？**我们将求解的结果回代回系统的状态表达式中：
$$
e^{\int_{t_0}^{t} A(\tau) \mathrm{d}\tau} = I + \int_{t_0}^{t} A(\tau) \mathrm{d}\tau + \frac{1}{2!} \left[ \int_{t_0}^{t} A(\tau) \mathrm{d}\tau \right]^2 + \cdots \\
\frac{\mathrm{d}}{\mathrm{d}t} e^{\int_{t_0}^{t} A(\tau) \mathrm{d}\tau} = A(t) + \int_{t_0}^{t} A(\tau) \mathrm{d}\tau A(t) + \frac{1}{2} \left[ \int_{t_0}^{t} A(\tau) \mathrm{d}\tau \right]^2 A(t) + \cdots \\
A(t)e^{\int_{t_0}^{t} A(\tau) \mathrm{d}\tau} = A(t) + A(t)\int_{t_0}^{t} A(\tau) \mathrm{d}\tau + A(t)\frac{1}{2!} \left[ \int_{t_0}^{t} A(\tau) \mathrm{d}\tau \right]^2 + \cdots \\
$$

显然，$\frac{\mathrm{d}}{\mathrm{d}t} e^{\int_{t_0}^{t} A(\tau) \mathrm{d}\tau} = A(t)e^{\int_{t_0}^{t} A(\tau) \mathrm{d}\tau}$ 的充分必要条件是 $ \int_{t_0}^{t} A(\tau) \mathrm{d}\tau A(t) = A(t)\int_{t_0}^{t} A(\tau) \mathrm{d}\tau$。

* 如果，$ \int_{t_0}^{t} A(\tau) \mathrm{d}\tau A(t) = A(t)\int_{t_0}^{t} A(\tau) \mathrm{d}\tau$
  
  $$
  x(t) = e^{\int_{t_0}^{t} A(\tau) \mathrm{d}\tau} x_0
  $$

* 如果，如果，$ \int_{t_0}^{t} A(\tau) \mathrm{d}\tau A(t) \ne A(t)\int_{t_0}^{t} A(\tau) \mathrm{d}\tau$
  
  $$
  x(t) = \left\{ I + \int_{t_0}^{t} A(\tau) \mathrm{d}\tau + \int_{t_0}^{t} A(\tau_1) \left[ \int_{t_0}^{\tau_1} A(\tau_2) \mathrm{d}\tau_2 \right] \mathrm{d}\tau_1 + \int_{t_0}^{t} A(\tau_1) \left[ \int_{t_0}^{\tau_1} A(\tau_2) \left[ \int_{t_0}^{\tau_2} A(\tau_3) \mathrm{d}\tau_3 \right] \mathrm{d}\tau_2 \right] \mathrm{d}\tau_1 + \cdots \right\} x(t_0)
  $$
  
  该式称为 Peano-Baker series

（**老何上课说，考试都会出可以交换的题目，所以重点记忆第一种情况**）



## 离散化

并不是所有的系统都是连续的，也存在离散信号系统。因此需要**将状态空间模型推广到离散域中**。

连续空间中的状态空间模型如下：

$$
\begin{cases}
\dot{x}(t) = Ax(t) + Bu(t) \\
y(t) = Cx(t) + Du(t)
\end{cases}
$$

* 对于**状态量**而言，假设**等间隔 $T$ 采样**，采样信号为：

$$
x^*(t) := \begin{cases} 
x(t), & t = kT, \\
0, & t \neq kT.
\end{cases}
$$

注意，这里的采样周期 $T$ 须符合奈奎斯特采样定理。

对于**控制量**而言，因为要做用于实际的 Plant，因此需要使用 D/A 转换器（ZOH），将控制器的离散输出转为连续量：

$$
u(t) = u(kT), \quad kT \le t \le (k+1)T
$$

接着，基于连续信号的状态转移方程：

$$
x(t) = \Phi(t-t_0)x(0) + \int_{t_0}^t \Phi(t-\tau) Bu(\tau) d\tau
$$

我们可以得到**离散信号的状态转移方程**：

$$
\begin{align}
x[(k+1)T] &= \Phi((k+1)T-kT)x(kT) + \int_{kT}^{(k+1)T} \Phi((k+1)T-\tau) Bu(\tau) d\tau \\
		  &= \Phi(T)x(kT) + \int_{kT}^{(k+1)T} \Phi((k+1)T-\tau) B d\tau u(kT) \\
		  &= e^{AT}x(kT) + \int_{0}^{T} e^{A\tau}  d\tau B u(kT)
\end{align}
$$

输出方程也同样经过信号采样：

$$
y(kT) = Cx(kT) + Du(kT)
$$

因此，得到**离散的状态空间模型**：

$$
\begin{cases}
x(k+1) = G(T)x(k) + H(T)u(k) \\
y(k) = Cx(k) + Du(k)
\end{cases}
$$

其中，$G(T) = e^{AT}$，$H(T) = \int_{0}^{T} e^{A\tau}  d\tau B$ 。

$$
\int_{0}^{T} e^{A\tau}  d\tau = \int_{0}^{T} (I + A\tau + \frac{\tau^2}{2!}A^2 + \cdots)  d\tau = TI + \frac{T^2}{2!}A + \frac{T^3}{3!}A^2 + \cdots \\
\text{如果 A 可逆，} \quad \int_{0}^{T} e^{A\tau} = A^{-1}(TA + \frac{T^2}{2!}A^2 + \frac{T^3}{3!}A^3 + \cdots) = A^{-1}(e^{AT} - I)
$$

同样的，尝试求解**离散状态空间模型的系统响应**（假设已知系统初始状态量 $x(0)$，以及每一时刻的控制 $u(k)$）：

$$
\begin{align}
&x(1) = Gx(0) + Hu(0) \\
&x(2) = Gx(1) + Hu(1) = G^2 x(0) + GHu(0) + Hu(1) \\
&x(3) = Gx(2) + Hu(2) = G^3 x(0) + G^2 Hu(0) + GHu(1) + Hu(2) \\
\vdots
\end{align}
$$

经过数学归纳，可以得到**系统响应**：

$$
x(k) = G^k x(0) + \sum_{i=0}^{k-1} G^{k-i-1} H u(i), \quad k = 1, 2, 3, \ldots, \\
y(k) = CG^k x(0) + C \sum_{i=0}^{k-1} G^{k-i-1} H u(i) + Du(k)
$$
