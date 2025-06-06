---
layout:       post
title:        "【现代控制理论】1-State Space Model"
author:       "Orchid"
header-style: text
catalog:      true
hidden:       false
tags:
    - 控制
    - notes
---

## Basic Concepts

现代控制理论相对于自动控制原理，在对于系统的描述中，引入了状态量（state variables），来进一步细节的描述系统。

* 系统量（System variables）：任何对于系统输入和初始状态有响应的变量
* **状态量（State variables）**：**线性独立的系统量的最小集合**，系统输入和初始状态可以通过这些变量完全确定系统输出

对于一个系统，其**状态量的数目**的唯一的，可以有以下几个条件确定：

* 描述系统的微分方程的阶数
* 描述系统的传递函数的分母的阶数
* 系统中独立的储能元件个数

由状态量，可以进一步得到以下几个概念：

* 状态向量（State vector）：由状态量组成的 $n \times 1$ 的列向量
* 状态空间（State space）：由状态量构成的 $n$ 维空间

以及，最最最重要的**状态空间方程（State-Space Equations）**，建立输入量、状态量、输出量之间的关系：

* **状态方程（State equation）**：$\dot{x}(t) = f(x,u,t)$，即状态量的变化与当前状态量、当前输入量及时间相关

* **输出方程（Output equation）**：$y(t) = g(x,u,t)$，即当前输出量与当前状态量、当前输入量及时间相关

  以下是最 general 的矩阵表现形式：
  
  $$
  \begin{align}
      \mathbf{x}(t) &= \begin{bmatrix}
      x_1(t) \\
      x_2(t) \\
      \vdots \\
      x_n(t)
      \end{bmatrix}, \quad
      \mathbf{f}(\mathbf{x}, \mathbf{u}, t) = \begin{bmatrix}
      f_1(x_1, x_2, \ldots, x_n; u_1, u_2, \ldots, u_r; t) \\
      f_2(x_1, x_2, \ldots, x_n; u_1, u_2, \ldots, u_r; t) \\
      \vdots \\
      f_n(x_1, x_2, \ldots, x_n; u_1, u_2, \ldots, u_r; t)
      \end{bmatrix}, \\
      \mathbf{y}(t) &= \begin{bmatrix}
      y_1(t) \\
      y_2(t) \\
      \vdots \\
      y_m(t)
      \end{bmatrix}, \quad
      \mathbf{g}(\mathbf{x}, \mathbf{u}, t) = \begin{bmatrix}
      g_1(x_1, x_2, \ldots, x_n; u_1, u_2, \ldots, u_r; t) \\
      g_2(x_1, x_2, \ldots, x_n; u_1, u_2, \ldots, u_r; t) \\
      \vdots \\
      g_m(x_1, x_2, \ldots, x_n; u_1, u_2, \ldots, u_r; t)
      \end{bmatrix}.
  \end{align}
  $$

接着，可以引入一些条件来简化对于状态空间的建模，即简化状态空间方程：

* 如果，状态空间方程关于状态量是线性的（注意，状态量间的线性的，但状态空间方程不一定线性），那么可以化简为：
  
  $$
  \begin{align}
  \dot{\mathbf{x}}(t) &= \mathbf{A}(t)\mathbf{x}(t) + \mathbf{B}(t)\mathbf{u}(t) \\
  \mathbf{y}(t) &= \mathbf{C}(t)\mathbf{x}(t) + \mathbf{D}(t)\mathbf{u}(t)
  \end{align}
  $$

* 如果，向量函数 $f$ 和 $g$ 与时间 $t$ 无关，或者矩阵 $A \  B \ C \ D$ 与时间 $t$ 无关，那么可以化简为：
  
  $$
  \begin{cases}
  \dot{x} = Ax + Bu \\
  y = Cx + Du
  \end{cases}
  $$
  
  其中，$A$ 为状态矩阵（state matrix），$B$ 为输入矩阵（input matrix），$C$ 为输出矩阵（output matrix），$D$ 为直接传递矩阵（direct transfer matrix）



## Establish State-space Model
那么如何根据一个实际的系统，来建立状态空间模型呢？首先，让我们回顾一下，在学习现代控制理论之前是如何描述一个实际系统的：在自动控制原理中，我们会使用**方框图**（Block Diagram Method）、**传递函数**（Transfer function method）；当然我们也可以根据物理学的知识，直接用**牛顿定律**或是**电路理论**来描述。那么，在这些已有的方法的基础上，我们很容易来建立状态空间模型。

### Block Diagram Method
我先直截了当的给出，基于方框图建立状态空间模型的流程：
1. 将方框图转化为标准形式，即由比例器、积分器、加法器构成的形式
2. 选择积分器的输出信号作为状态变量
3. 根据方框图的连接，写出状态变量相关的方程等式
4. 将上述方程等式转换成标准的状态方程和输出方程

**Example**
Step 1 & Step 2：

<img src="https://notes.sjtu.edu.cn/uploads/upload_113337735bf54a1cdafe5f9a4f9f0961.png" style="zoom:67%;" />

<img src="https://notes.sjtu.edu.cn/uploads/upload_baffe82f9909c8150060e03e27ad9739.png" style="zoom:67%;" />

Step 3：

<img src="https://notes.sjtu.edu.cn/uploads/upload_a90761a3b0dc2045c68b04ac156e56e8.png" style="zoom: 67%;" />

Step 4：

<img src="https://notes.sjtu.edu.cn/uploads/upload_2792627a9a1caa282ba1fd28184c8fdd.png" style="zoom:67%;" />

### Physical Principles Analysis
同样，我先给出基于物理分析，建立状态空间模型的流程：
1. 选择合适的物理变量作为状态变量
2. 基于物理分析，建立状态变量之间的方程
3. 转换为标准形式

在步骤一中，有一个小技巧。因为我们知道，对于一个系统，其状态量的数目等于系统中独立的储能元件个数，因此**可以选择与独立储能元件相关的物理量作为状态变量**，比如电容（U）、电感（I）、弹簧（x）、阻尼（v）。

**Example**

<img src="https://notes.sjtu.edu.cn/uploads/upload_198c0d8d95ad96dffab0090fa9e5c7e8.png" style="zoom:67%;" />

Step 1： Choosing state variables，$x_1 = i_L; \quad x_2 = u_c$

Step 2:

$$
\begin{align*}
& i_L = (u - L \frac{di_L}{dt}) \frac{1}{R_1} + C \frac{du_C}{dt} \\
& L \frac{di_L}{dt} + C \frac{du_C}{dt} R_2 + u_C = u
\end{align*}
$$

Step 3:

$$
\begin{align*}
\begin{bmatrix}
\dot{x}_1 \\
\dot{x}_2
\end{bmatrix}
&=
\begin{bmatrix}
-\frac{1}{L}\frac{R_1 R_2}{R_1 + R_2} & -\frac{R_1}{L(R_1 + R_2)} \\
\frac{R}{C(R_1 + R_2)} & -\frac{1}{C(R_1 + R_2)}
\end{bmatrix}
\begin{bmatrix}
x_1 \\
x_2
\end{bmatrix}
+
\begin{bmatrix}
\frac{1}{L} \\
0
\end{bmatrix}
u \\
y &= \begin{bmatrix} 0 & 1 \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}
\end{align*}
$$

### Transfer function method
基于传递函数，建立状态空间模型，较前两种方法而言，其更加抽象而复杂。因此我们先**明确以下目标**，我们需要基于微分方程

$$
y^{(n)} + a_{n-1}y^{(n-1)} + \cdots + a_1y^{(1)} + a_0y = b_mu^{(m)} + b_{m-1}u^{(m-1)} + \cdots + b_1u^{(1)} + b_0u
$$

或传递函数

$$
W(s) = \frac{Y(s)}{U(s)} = \frac{b_m s^m + b_{m-1} s^{m-1} + \cdots + b_1 s + b_0}{s^n + a_{n-1} s^{n-1} + \cdots + a_1 s + a_0} \quad m \leq n
$$

建立状态空间方程

$$
\begin{cases}
  \dot{x} = Ax + Bu \\
  y = Cx + Du
\end{cases}
$$

注意，为使得系统有实际的意义，需要保证 $n \geq m$

#### Easy case

OK，接下来进入正题，首先考虑一个简单的传递函数（微分方程）：

$$
\frac{d^n y}{dt^n} + a_{n-1} \frac{d^{n-1} y}{dt^{n-1}} + \cdots + \frac{dy}{dt} + a_0 y = b_0 u
$$

我们知道系统状态量的数目等于系统的微分方程的阶数，因此可以选择 n 个状态量：

$$
\begin{align*}
x_1 &= y \\
x_2 &= \dot{y} \Rightarrow \dot{x}_1 = x_2 \\
&\vdots \\
x_n &= \frac{d^{n-1}y}{dt^{n-1}} \Rightarrow \dot{x}_{n-1} = x_n \\
\dot{x}_n &= \frac{d^n y}{dt^n} = -a_0x_1 - a_1x_2 \cdots - a_{n-1}x_n + b_0u
\end{align*}
$$

然后，我们就可以将上述式子整理成标准的状态空间方程：

$$
\frac{d}{dt}
\begin{bmatrix}
x_1 \\
x_2 \\
\vdots \\
x_{n-1} \\
x_n
\end{bmatrix}
=
\begin{bmatrix}
0 & 1 & 0 & \cdots & 0 \\
0 & 0 & 1 & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & \cdots & 1 \\
-a_0 & -a_0 & -a_0 & \cdots & -a_0
\end{bmatrix}
\begin{bmatrix}
x_1 \\
x_2 \\
\vdots \\
x_{n-1} \\
x_n
\end{bmatrix}
+
\begin{bmatrix}
0 \\
0 \\
\vdots \\
0 \\
b_0
\end{bmatrix}
\\

y =
\begin{bmatrix}
1 & 0 & 0 & \cdots & 0
\end{bmatrix}
\begin{bmatrix}
x_1 \\
x_2 \\
\vdots \\
x_{n-1} \\
x_n
\end{bmatrix}
$$

#### Complex case

接着讨论一下复杂的情形：

$$
g(s) = \frac{\beta_1 s^{n-1} + \beta_2 s^{n-2} + \cdots + \beta_{n-1}s + \beta_n}{s^n + \alpha_1 s^{n-1} + \cdots + \alpha_{n-1}s + \alpha_n}
$$

我们可以将这个传递函数按照下图，分解为两个部分：

<img src="https://notes.sjtu.edu.cn/uploads/upload_4591784576858eee487ca84eb1cc5240.png" style="zoom:67%;" />

首先，看左边**第一部分**的传递函数：

$$
\begin{align}
&\frac{z(s)}{u(s)} = \frac{1}{s^n + a_{n-1}s^{n-1} + \cdots + a_1s + a_0} \\
\Rightarrow &z^{(n)} + a_{n-1}z^{(n-1)} + \cdots + a_1\dot{z} + a_0z = u
\end{align}
$$

然后，看右边**第二部分**的传递函数：

$$
\begin{align}
&\frac{y(s)}{z(s)} = \beta_{n-1}s^{n-1} + \cdots + \beta_1s + \beta_0 \\
\Rightarrow &y = \beta_{n-1}z^{(n-1)} + \cdots + \beta_1\dot{z} + \beta_0z
\end{align}
$$

因此可以如下选择状态量，并建立方程：

$$
x_1 = z, x_2 = \dot{z}, x_3 = \ddot{z}, \cdots, x_n = z^{(n-1)}\\
$$

$$
\dot{x}_1 = x_2 \\
\dot{x}_2 = x_3 \\
\vdots \\
\dot{x}_n = -a_0z - a_1\dot{z} - \cdots - a_{n-1}z^{(n-1)} + u \\
= -a_0x_1 - a_1x_2 - \cdots - a_{n-1}x_n + u
$$

$$
y = \beta_0x_1 + \beta_1x_2 + \cdots + \beta_{n-1}x_n
$$

最后可以得到状态方程：

$$
\dot{x} = \left[\begin{array}{ccccc}
0 & 1 & 0 & \cdots & 0 \\
0 & 0 & 1 & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & \cdots & 1 \\
-a_0 & -a_1 & -a_2 & \cdots & -a_{n-1}
\end{array}\right] x + \left[\begin{array}{c}
0 \\
0 \\
\vdots \\
0 \\
1
\end{array}\right] u\\

y = \left[\begin{array}{cccc}
b_0 & b_1 & \cdots & b_{n-1}
\end{array}\right] x + d u
$$

该形式的状态空间模型被称为 **“controllable canonical form”**（**非常非常重要啊！！！**）

OK，上面已经给出了基于传递函数（微分方程）建立状态空间模型的方法。到这儿就结束了吗？NO，no，no！在上述的 Complex case 中，其实已经埋下了伏笔。我们是如何应对 Complex case 的，将传递函数分为分子和分母两部分处理。那么能不能有其他不同的分解方式，将复杂的传递函数分解为简单传递函数的组合呢？Of course！

#### Tandem decomposition 串联分解

$$
\frac{Y(s)}{U(s)} = b_n \cdot \frac{s + z_1}{s + p_1} \cdot \frac{s + z_2}{s + p_2} \cdots \frac{s + z_n}{s + p_n}, (m = n)\\

\frac{Y(s)}{U(s)} = b_n \cdot \frac{s + z_1}{s + p_1} \cdot \frac{s + z_2}{s + p_2} \cdots \frac{s + z_m}{s + p_m} \cdot \frac{1}{s + p_{m+1}} \cdots \frac{1}{s + p_n}, (m \leq n)
$$

<img src="https://notes.sjtu.edu.cn/uploads/upload_25810d2cf33bffa1ee4b3546f590bfc2.png" style="zoom:67%;" />

接着，让我们看看每一项是如何转换的：

**case 1：**

$$
\frac{Y_i(s)}{Y_{i-1}(s)} = \frac{1}{s + p_i} = \frac{\frac{1}{s}}{1 + \frac{p_i}{s}}
$$

<img src="https://notes.sjtu.edu.cn/uploads/upload_24ead8dc00e7a934d60a6d32550d272d.png" style="zoom:67%;" />

**case 2：**

$$
\frac{Y_i(s)}{Y_{i-1}(s)} = \frac{s + z_i}{s + p_i} = 1 + \frac{z_i - p_i}{s + p_i}
$$

<img src="https://notes.sjtu.edu.cn/uploads/upload_373dd5ea3cf274a747fa0cba4e2447d3.png" style="zoom:75%;" />

注意，case 2 中的 $z_i - p_i$ 可以放在前面（对输入放缩）也可以放在后面（对输出放缩）。

**Example：**

$$
\frac{Y(s)}{U(s)} = \frac{b_1(s - z_2)(s - z_3)}{(s - p_1)(s - p_2)(s - p_3)}
$$

<img src="https://notes.sjtu.edu.cn/uploads/upload_d902f8ced44368d08ce49a75d221984d.jpg" style="zoom:50%;" />

$$
\dot{x} = \left[\begin{array}{ccc}
p_1 & 0 & 0 \\
1 & p_2 & 0 \\
1 & p_2 - z_2 & p_3
\end{array}\right] x + \left[\begin{array}{c}
b_1 \\
0 \\
0
\end{array}\right] u\\
y = \left[\begin{array}{ccc}
1 & p_2 - z_2 & p_3 - z_3
\end{array}\right] x
$$

#### Parallel decomposition 并联分解

$$
\frac{Y(s)}{U(s)} = \frac{b_m s^m + b_{m-1} s^{m-1} + \cdots + b_1 s + b_0}{s^n + a_{n-1} s^{n-1} + \cdots + a_1 s + a_0} \\
\Rightarrow \frac{Y(s)}{U(s)} = \sum_{j=1}^{n} \frac{c_j}{s + p_j} + b_n \ (No \ repeated \ poles)\\
\Rightarrow \frac{Y(s)}{U(s)} = \sum_{i=1}^{k} \frac{c_i}{(s+p_1)^{k-i+1}} + \sum_{j=k+1}^{n} \frac{c_j}{s+p_j} + b_n \ (With \ repeated \ poles)
$$

**No repeated poles：**

$$
X_i(s) = \frac{1}{s - p_i} U(s), \quad i = 1, 2, \cdots, n \\
\dot{x}_i(t) = p_i x_i(t) + u(t), \quad i = 1, 2, \cdots, n \\
y(t) = c_1 x_1(t) + c_2 x_2(t) + \cdots + c_n x_n(t) \\
\Rightarrow
\begin{bmatrix}
\dot{x}_1 \\
\vdots \\
\dot{x}_n
\end{bmatrix}
=
\begin{bmatrix}
p_1 & 0 & \cdots & 0 \\
0 & p_2 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & p_n
\end{bmatrix}
\begin{bmatrix}
x_1 \\
x_2 \\
\vdots \\
x_n
\end{bmatrix}
+
\begin{bmatrix}
1 \\
1 \\
\vdots \\
1
\end{bmatrix}
u \\
y = \left[\begin{array}{cccc}
c_1 & c_2 & \cdots & c_n
\end{array}\right] x
$$

**With repeated poles：**

选择状态量：

$$
\left\{
\begin{aligned}
X_1(s) &= \frac{U(s)}{(s - p_1)^r} \\
X_2(s) &= \frac{U(s)}{(s - p_1)^{r-1}} \\
&\vdots \\
X_r(s) &= \frac{U(s)}{s - p_1} \\
X_{r+1}(s) &= \frac{U(s)}{s - p_{r+1}} \\
&\vdots \\
X_n(s) &= \frac{U(s)}{s - p_n}
\end{aligned}
\right.
$$

列出状态量的关系式：

$$
\left\{
\begin{aligned}
X_1(s) &= \frac{1}{s - p_1} X_2(s) \\
X_2(s) &= \frac{1}{s - p_1} X_3(s) \\
&\vdots \\
X_{r-1}(s) &= \frac{1}{s - p_1} X_r(s) \\
X_r(s) &= \frac{1}{s - p_1} U(s) \\
X_{r+1}(s) &= \frac{1}{s - p_{r+1}} U(s) \\
&\vdots \\
X_n(s) &= \frac{1}{s - p_n} U(s)
\end{aligned}
\right.
$$

$$
\left\{
\begin{aligned}
\dot{x}_1 &= p_1 x_1 + x_2 \\
\dot{x}_2 &= p_1 x_2 + x_3 \\
&\vdots \\
\dot{x}_r &= p_1 x_r + u \\
\dot{x}_{r+1} &= p_{r+1} x_{r+1} + u \\
&\vdots \\
\dot{x}_n &= p_n x_n + u
\end{aligned}
\right. \\
y(t) = c_{11}x_1(t) + c_{12}x_2(t) + \cdots + c_{1r}x_r(t) + c_{r+1}x_{r+1}(t) + \cdots + c_nx_n(t) 
$$

转换成标准形式：

$$
\begin{bmatrix}
\dot{x}_1 \\
\dot{x}_2 \\
\vdots \\
\dot{x}_r \\
\dot{x}_{r+1} \\
\vdots \\
\dot{x}_n
\end{bmatrix}
=
\begin{bmatrix}
p_1 & 1 & 0 & \cdots & 0 & \cdots & 0 \\
0 & p_1 & 1 & \cdots & 0 & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots & \vdots & \vdots \\
0 & 0 & 0 & \cdots & p_1 & \cdots & 0 \\
0 & 0 & 0 & 0 & \cdots & p_{r+1} & \cdots  \\
\vdots & \vdots & \vdots & \ddots & \vdots & \vdots & \vdots \\
0 & 0 & 0 & \cdots & 0 & 0 & p_n
\end{bmatrix}
\begin{bmatrix}
x_1 \\
x_2 \\
\vdots \\
x_r \\
x_{r+1} \\
\vdots \\
x_n
\end{bmatrix}
+
\begin{bmatrix}
0 \\
0 \\
\vdots \\
1 \\
\vdots \\
1
\end{bmatrix}
u
$$

$$
y = \left[\begin{array}{cccc}
c_{11} & c_{12} & \cdots & c_{1r} & c_{r+1} & \cdots &c_n
\end{array}\right] x
$$

### From SSM to G(s)

我们现在已经知道，如何基于传递函数来建立状态空间模型，那么能否反过来，已知状态空间模型来反推系统的传递函数呢？答案是肯定的。下面以 SISO system 为例：

$$
\begin{cases}
  \dot{x} = Ax + Bu \\
  y = Cx + Du
  \end{cases}
$$

通过拉氏变换，可以得到：

$$
sX(s) - x(0) = AX(s) + Bu(s) \\
Y(s) = CX(s) + Du(s)
$$

通常，我们**假设系统的初始状态量为0**，那么可以得到：

$$
y(s)=[C(sI-A)^{-1}B+D]u(s)
$$

所以，可以得到传递函数：

$$
G(s) = \frac{y(s)}{u(s)} = C(sI-A)^{-1}B+D = C\frac{adj(sI-A)}{|sI-A|}B + D
$$

## Composite System

#### Parallel connection

$$
\Sigma_1:
\left\{
\begin{aligned}
\dot{x}_1 &= A_1x_1 + B_1u_1 \\
y_1 &= C_1x_1 + D_1u_1
\end{aligned}
\right.
\\
\Sigma_2:
\left\{
\begin{aligned}
\dot{x}_2 &= A_2x_2 + B_2u_2 \\
y_2 &= C_2x_2 + D_2u_2
\end{aligned}
\right.
$$

<img src="https://notes.sjtu.edu.cn/uploads/upload_6ca5f18061f585805b06dbc411250c50.png" style="zoom:67%;" />

SSM：

$$
\begin{bmatrix}
\dot{x}_1 \\
\dot{x}_2
\end{bmatrix}
=
\begin{bmatrix}
A_1 & 0 \\
0 & A_2
\end{bmatrix}
\begin{bmatrix}
x_1 \\
x_2
\end{bmatrix}
+
\begin{bmatrix}
B_1 \\
B_2
\end{bmatrix}
u
$$

$$
\begin{align}
y &= y_1 + y_2 \\
&= (C_1x_1 + C_2x_2) + (D_1u + D_2u) \\
&= \begin{pmatrix} C_1 & C_2 \end{pmatrix} \begin{pmatrix} x_1 & x_2 \end{pmatrix}^T + (D_1 + D_2)u
\end{align}
$$

G(s)：

$$
G(s) = G_1(s) + G_2(s)
$$

####  Series connection

$$
\Sigma_1:\left\{\begin{aligned}\dot{x}_1 &= A_1x_1 + B_1u_1 \\y_1 &= C_1x_1 + D_1u_1\end{aligned}\right.\\\Sigma_2:\left\{\begin{aligned}\dot{x}_2 &= A_2x_2 + B_2u_2 \\y_2 &= C_2x_2 + D_2u_2\end{aligned}\right.
$$

<img src="https://notes.sjtu.edu.cn/uploads/upload_2d01a1b93df6e38ff97cfc2d3d2cd90e.png" style="zoom:67%;" />

$$
\dot{x}_1 = A_1x_1 + B_1u_1 \\
\dot{x}_2 = A_2x_2 + B_2y_1 = A_2x_2 + B_2(C_1x_1 + D_1u) \\
y_2 = C_2x_2 + D_2y_1 = C_2x_2 + D_2(C_1x_1 + D_1u) \\
$$

SSM：

$$
\begin{bmatrix}
\dot{x}_1 \\
\dot{x}_2
\end{bmatrix}
=
\begin{bmatrix}
A_1 & 0 \\
B_2C_1 & A_2
\end{bmatrix}
\begin{bmatrix}
x_1 \\
x_2
\end{bmatrix}
+
\begin{bmatrix}
B_1 \\
B_2D_1
\end{bmatrix}
u \\
$$

$$
y = \begin{bmatrix}
D_2C_1 & C_2
\end{bmatrix}
\begin{bmatrix}
x_1 \\
x_2
\end{bmatrix}
+ D_2D_1u
$$

G(s)：

$$
G(s) = G_2(s)G_1(s)
$$

####  Feedback

$$
\Sigma_1:\left\{\begin{aligned}\dot{x}_1 &= A_1x_1 + B_1u_1 \\y_1 &= C_1x_1 \end{aligned}\right.\\\Sigma_2:\left\{\begin{aligned}\dot{x}_2 &= A_2x_2 + B_2u_2 \\y_2 &= C_2x_2 \end{aligned}\right.
$$

<img src="https://notes.sjtu.edu.cn/uploads/upload_af5ffeb30c35d119170f6354533cffed.png" style="zoom:67%;" />

$$
\dot{x}_1 = A_1x_1 + B_1u_1 = A_1x_1 - B_1C_2x_2 + B_1u \\
\dot{x}_2 = A_2x_2 + B_2u_2 = A_2x_2 + B_2C_1x_1 \\
y = y_1 = C_1x_1 \\
$$

SSM：

$$
\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} = \begin{bmatrix} A_1 & -B_1C_2 \\ B_2C_1 & A_2 \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} + \begin{bmatrix} B_1 \\ 0 \end{bmatrix} u \\
y = \begin{bmatrix} C_1 & 0 \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}
$$

G(s)：

$$
G(s) = [I+G_1(s)G_2(s)]^{-1}G_1(s)
$$


## Linear Transformation

我们知道，使用状态空间模型去描述一个系统，其状态量的选择不是唯一的（状态量的数目是唯一的）。那么可以考虑，$x=[x_1,x_2,\cdots,x_n]^T$ 是一组由 n 个状态量构成的 n 维状态向量，则 $x_1,x_2,\cdots,x_n$ 的**线性组合** $\bar{x}_1,\bar{x}_2,\cdots,\bar{x}_n$ 也可以作为一组新的状态量。因为状态量是线性独立的系统量的最小集合，那么可以说，这两组状态量之间存在着**非奇异线性变换关系**：

$$
x = P \bar{x}
$$

其中，$P$ 是 $n \times n$ 的非奇异变换矩阵。

下面讨论状态量线性变换后，状态空间表达式的变化：

$$
\begin{cases}
  \dot{x} = Ax + Bu \\
  y = Cx + Du
\end{cases} \quad
\Rightarrow \quad
\begin{cases}
  \dot{\bar{x}} = P^{-1}AP\bar{x} + P^{-1}Bu \\
  y = CP\bar{x} + Du
\end{cases}
$$

因此，可以令

$$
\begin{align}
\bar{A} &= P^{-1}AP \\
\bar{B} &= P^{-1}B \\
\bar{C} &= CP \\
\bar{D} &= D
\end{align}
$$

### 对角标准型

通过上述讨论，我们知道可以通过线性变换，将一组状态量转换为一组新的状态量，同时得到新的状态空间表达式。很显然会想，为什么要线性变换？是不是通过线性变换得到的新的状态空间表达式能有一些好的性质呢？答案显然如此。我们所希望的是：**如果能将状态量解耦，即状态矩阵 $A$ 是一个对角阵，则状态空间模型将大大简化**。而线性变换恰恰是实现这一目标的工具。

这里就已经是一个数学问题了。而事实上并非所有的状态矩阵 $A$ 能通过 $P^{-1}AP$ 转换为一个对角阵，这里要做一些数学上的假设：状态矩阵 $A_{n\times n}$ 有 n 个两两互异的特征值 $\lambda_1, \lambda_2, \cdots, \lambda_n$，则每个特征值都对应一个特征向量，$v_1, v_2, \cdots, v_n$ ，即 $Av_i = \lambda_i v_i$。我们可以得到，由特征向量组成的变换阵 $P = [v_1, v_2, \cdots, v_n]$ 能将 $\bar{A} = P^{-1}AP$ 转换为一个对角阵。

 证明：

$$
\begin{align}
AP &= [Av_1, Av_2, \cdots, Av_n] \\
   &= [\lambda_1v_1, \lambda_2v_2, \cdots, \lambda_nv_n] \\
   &= \begin{bmatrix}
   \lambda_1 & 0 & \cdots & 0 \\
   0 & \lambda_2 & \cdots & 0 \\
   \vdots & \vdots & \ddots & \vdots \\
   0 & 0 & \cdots & \lambda_n
   \end{bmatrix}
   \begin{bmatrix}
   v_1, v_2, \cdots, v_n
   \end{bmatrix} \\
  &= \begin{bmatrix}
   v_1, v_2, \cdots, v_n
   \end{bmatrix}
  	\begin{bmatrix}
   \lambda_1 & 0 & \cdots & 0 \\
   0 & \lambda_2 & \cdots & 0 \\
   \vdots & \vdots & \ddots & \vdots \\
   0 & 0 & \cdots & \lambda_n
   \end{bmatrix} \\
\Rightarrow P^{-1}AP &= [v_1, v_2, \cdots, v_n]^{-1} [v_1, v_2, \cdots, v_n] \begin{bmatrix}
   \lambda_1 & 0 & \cdots & 0 \\
   0 & \lambda_2 & \cdots & 0 \\
   \vdots & \vdots & \ddots & \vdots \\
   0 & 0 & \cdots & \lambda_n
   \end{bmatrix} \\
   &= \begin{bmatrix}
   \lambda_1 & 0 & \cdots & 0 \\
   0 & \lambda_2 & \cdots & 0 \\
   \vdots & \vdots & \ddots & \vdots \\
   0 & 0 & \cdots & \lambda_n
   \end{bmatrix} \\
   &= \bar{A}
\end{align}
$$


 特别的，当状态矩阵 $A$ 为标准形式：

$$
A = \begin{bmatrix}
0 & 1 & 0 & \cdots & 0  \\
0 & 0 & 1 & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & \cdots & 1 \\
-a_0 & -a_{1} & -a_{2} & \cdots & -a_{n-1}
\end{bmatrix}
$$

且有 n 个两两互异的特征值 $\lambda_1, \lambda_2, \cdots, \lambda_n$，则将状态矩阵转为对角阵的线性变换矩阵可以表示为：

$$
P = \begin{bmatrix}
1 & 1 & \cdots & 1 \\
\lambda_1 & \lambda_2 & \cdots & \lambda_n \\
\lambda_1^2 & \lambda_2^2 & \cdots & \lambda_n^2 \\
\vdots & \vdots & \ddots & \vdots \\
\lambda_1^{n-1} & \lambda_2^{n-1} & \cdots & \lambda_n^{n-1}
\end{bmatrix}
$$

### Jordan 标准型

在上一小节中，我们尝试使用线性变换，将状态空间表达式中的状态矩阵 $A$ 变换为对角阵 $\bar{A}$ 。但我们使用了非常强的数学假设，状态矩阵 $A_{n\times n}$ 有 n 个两两互异的特征值 $\lambda_1, \lambda_2, \cdots, \lambda_n$，则每个特征值都对应一个特征向量，$v_1, v_2, \cdots, v_n$。实际上，并不是所有的状态矩阵 $A_{n\times n}$ 都有 n 个两两互异的特征值，而往往有**重特征值**。以下需要分类讨论：

1. 重特征值的几何重数等于代数重数

   此时，仍然能够找到 n 个线性无关的特征向量来构成线性变换阵 $P$ （不同特征值的特征向量是线性无关的，同一特征值的不同特征向量也是线性无关的），使得 $\bar{A} = P^{-1}AP$ 变换得到的矩阵是对角阵，证明如上一小节。

2. 重特征值的几何重数小于代数重数（几何重数=1）

   此时，因为存在重特征值的几何重数小于代数重数，无法找到 n 个线性无关的征向量来构成线性变换阵 $P$ ，需要补充**广义特征向量**。广义特征值的求解方法如下：

   对于重特征值 $\lambda_i$，我们想要找到一组广义特征值 $v_i^{'}, v_i^{''},\cdots,v_i^{\sigma_i}$，使得：
   
   $$
   \begin{bmatrix}
   \mathbf{v}_i' & \mathbf{v}_i'' & \cdots & \mathbf{v}_i^{\sigma_i}
   \end{bmatrix}
   \begin{bmatrix}
   \lambda_i & 1 & & 0 \\
   & \lambda_i & \ddots & \\
   & & \ddots & 1 \\
   0 & & & \lambda_i
   \end{bmatrix}
   =
   A
   \begin{bmatrix}
   \mathbf{v}_i' & \mathbf{v}_i'' & \cdots & \mathbf{v}_i^{\sigma_i}
   \end{bmatrix}
   $$
   
   而 $v_i^{'}, v_i^{''},\cdots,v_i^{\sigma_i}$ 可以由以下式子确定：
   
   $$
   \left\{
   \begin{aligned}
   (\lambda_i I - A) \mathbf{v}_i' &= \mathbf{0} \\
   (\lambda_i I - A) \mathbf{v}_i'' &= -\mathbf{v}_i' \\
   (\lambda_i I - A) \mathbf{v}_i''' &= -\mathbf{v}_i'' \\
   &\vdots \\
   (\lambda_i I - A) \mathbf{v}_i^{\sigma_i} &= -\mathbf{v}_i^{(\sigma_i - 1)}
   \end{aligned}
   \right.
   $$
   

### Properties

1. **传递函数的不变性**

   已知，原系统的传递函数表达式为：
   
   $$
   G(s) = C(sI-A)^{-1}B + D
   $$
   
   变换后的系统的传递函数表达式为：
   
   $$
   \begin{align}
   \bar{G} &= \bar{C}(sI-\bar{A})^{-1}\bar{B} + \bar{D} \\
   		&= CP(sI-P^{-1}AP)^{-1}P^{-1}B + D \\
   		&= C[P(sI-P^{-1}AP)P^{-1}]^{-1} + D \\
   		&= C(sI-A)^{-1}B + D \\
   		&= G(s)
   \end{align}
   $$

2. **状态矩阵的特征值不变性**
   $$
   \begin{align}
   det(sI - \bar{A}) &= det(sI-P^{-1}AP) \\ 
   				  &= det(sP^{-1}P-P^{-1}AP) \\
   				  &= det(P^{-1})det(sI-A)det(P) \\
   				  &= det(sI-A)
   \end{align}
   $$

3. **同一系统不同状态向量之间必然存在一种线性变换的关系**
