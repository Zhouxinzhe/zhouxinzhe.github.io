---
layout:       post
title:        "【计算机控制技术】- Pole Placement Design"
author:       "Orchid"
header-style: text
catalog:      true
hidden:       false
tags:
    - 控制	
    - notes
---

上一章介绍了 Emulation Method for Digital Controller Design，主要的设计思路是先在连续时间系统中设计符合要求的模拟控制器，再将其以不同方式离散化成数字控制器。这种 Emulation 的思想是非常好的，但是也存在很多缺陷：对采样频率要求高（至少是信号带宽的20倍）、离散控制器存在时滞问题（Step-invariance method 的 ZOH 近似）……

因此，需要直接在离散域下设计数字控制器的方法！（*一般来说，采样频率小于10倍信号带宽时需要直接设计*）

怎么设计？首先得知道离散域下的对象建模，其次你要知道自己的设计目标是什么。这一章将从离散域的状态空间建模出发，通过极点配置（Pole Placement）的方法，实现系统的李雅普诺夫稳定，以及状态观测的收敛。

* 离散域建模：状态空间模型
* 设计目标：设计控制器，使得系统的李雅普诺夫稳定；设计观测器，使得状态观测的收敛



## Discretization of Continuous-time System

在上一章的最后一节，已经介绍了如何将连续时间系统的状态空间模型离散化为离散系统的状态空间模型，一般来说我们会选择**连续状态空间离散化的精确方法**：

$$
\text{if} \quad \mathbf{u}(t) = \mathbf{u}(kT), \quad kT \leq t < (k + 1)T \\
\begin{cases}
\mathbf{\dot{x}}(t) &= \mathbf{A_a}\mathbf{x}(t) + \mathbf{B_a}\mathbf{u}(t) \\
\mathbf{y}(t) &= \mathbf{C_a}\mathbf{x}(t) + \mathbf{D_a}\mathbf{u}(t)
\end{cases}
\quad \Rightarrow \quad
\begin{cases}
\mathbf{x}(k + 1) &= \mathbf{A}\mathbf{x}(k) + \mathbf{B}\mathbf{u}(k) \\
\mathbf{y}(k) &= \mathbf{C}\mathbf{x}(k) + \mathbf{D}\mathbf{u}(k)
\end{cases}
$$

$$
\begin{aligned}
\mathbf{A} &= \boldsymbol{\Phi}(T) = e^{AT}, \quad \mathbf{B} = \int_{0}^{T} \boldsymbol{\Phi}(\tau) d\tau \mathbf{B}_a, \quad \mathbf{C} = \mathbf{C}_a, \quad \mathbf{D} = \mathbf{D}_a
\end{aligned}
$$

如此，我们就得到了离散域下的状态空间模型。



## State feedback  

（这一部分和现代控制理论的状态反馈是一致的）

直接离散化得到的状态空间模型，本身可能不是李雅普诺夫稳定的（$\mathbf{A}$ 的特征值不一定小于 1），因此需要设计控制器使得系统稳定。本节介绍的方法是**基于反馈的控制器设计**。一般来说存在以下两种反馈：

* **State feedback**

  ![](https://notes.sjtu.edu.cn/uploads/upload_de16183ebccc58cddee7b06d2428689b.png)
  
  $$
  \mathbf{u}(k) = -\mathbf{K}\mathbf{x}(k) + \mathbf{N}\mathbf{r}(k)
  $$

* **Output feedback**

  ![](https://notes.sjtu.edu.cn/uploads/upload_7a5072f2348a9021e7cfbc623c969fe9.png)
  
  $$
  \mathbf{u}(k) = -\mathbf{K_y}\mathbf{y}(k) + \mathbf{N_y}\mathbf{r}(k)
  $$

状态反馈相对于输出反馈是直接反馈系统状态量，因此反馈的信息更多更全，效果也会更好。因此接下来都是对于**状态反馈**的分析。



对于以下系统，当控制器设计为状态反馈后，可以得到：

$$
\begin{cases}
\mathbf{x}(k + 1) &= \mathbf{A}\mathbf{x}(k) + \mathbf{B}\mathbf{u}(k) \\
\mathbf{y}(k) &= \mathbf{C}\mathbf{x}(k)
\end{cases}
\quad \Rightarrow \quad
\begin{cases}
\mathbf{x}(k + 1) &= [\mathbf{A-BK}]\mathbf{x}(k) + \mathbf{BN}\mathbf{r}(k) \\
\mathbf{y}(k) &= \mathbf{C}\mathbf{x}(k)
\end{cases}
$$

因此，可以通过设计 $\mathbf{K}$ 来设计系统 $[\mathbf{A-BK}]$ 的特征值，来实现系统的稳定性。

同时引入状态反馈会改变系统的稳态增益 

$$
\lim_{z\rightarrow 1} (z-1)Y(z) = \lim_{z\rightarrow 1} (z-1)R(z)G(z) = \lim_{z\rightarrow 1} z\mathbf{C}[z\mathbf{I} - \mathbf{A+BK}]^{-1}\mathbf{BN}
$$
因此可以通过设计 $\mathbf{N}$ 来保证系统的稳态增益。

* 设计 $\mathbf{K}$ 来实现系统的稳定性（*主要关注点*）
* 设计 $\mathbf{N}$ 来保证系统的稳态增益



接下来是一些设计 $\mathbf{K}$ 的方法：

* 直接求解法（*一般用这种*）：

  * 人为确定系统的特征值（转递函数的极点），得到特征方程 $\alpha_c(z)$
  * $\|\mathbf{A-BK}\| = \alpha_c(z)$ 直接求解 $\mathbf{K}$

* 能控标准型法：

  * 先将系统线性变换成能控标准型，轻松得到变换后的系统下的 $\mathbf{K_w}$
    
    $$
    \mathbf{\bar{A}} - \mathbf{\bar{B}K_w} = \begin{bmatrix}
    0 & 1 & 0 & \cdots & 0 \\
    0 & 0 & 1 & \cdots & 0 \\
    0 & 0 & 0 & \cdots & 0 \\
    \vdots & \vdots & \vdots & & \vdots \\
    0 & 0 & 0 & \cdots & 1 \\
    -(a_0 + K_1) & -(a_1 + K_2) & -(a_2 + K_3) & \cdots & -(a_{n-1} + K_n)
    \end{bmatrix}
    $$

    $$
    \text{let} \quad \alpha_c(z) = z^n + (a_{n-1} + K_{wn})z^{n-1} + \cdots + (a_1 + K_{w2})z + (a_0 + K_{w1}) \\
    \text{then} \quad K_{w1} = \alpha_0 - a_0, \quad K_{w2} = \alpha_1 - a_1, \quad \cdots \quad K_{wn} = \alpha_{n-1} - a_{n-1}
    $$

    $$
    \mathbf{K = K_w P}
    $$

* Ackermann’s formula  
  
  $$
  \text{desired characteristic equation:}\quad \alpha_c(z) = z^n + \alpha_{n-1}z^{n-1} + \cdots + \alpha_1z + \alpha_0 = 0 \\
  \text{Form the matrix polynomial:} \quad \alpha_c(\mathbf{A}) = \mathbf{A}^n + \alpha_{n-1}\mathbf{A}^{n-1} + \cdots + \alpha_1\mathbf{A} + \alpha_0\mathbf{I}
  $$

  $$
  \mathbf{K} = [0 \ 0 \ \cdots \ 0 \ 1] \begin{bmatrix} \mathbf{B} & \mathbf{AB} & \cdots & \mathbf{A}^{n-2}\mathbf{B} & \mathbf{A}^{n-1}\mathbf{B} \end{bmatrix}^{-1} \alpha_c(\mathbf{A})
  $$

从上述的第二种方法也可以看到，**通过状态反馈来实现系统稳定的前提条件是，系统是能控的！**



## State Observer

为什么在控制器设计章节讲观测器？因为在上一小节，我们所设计的控制器是基于状态反馈的，必须知道系统的状态量才能够实现的闭环控制器。但事实上是，在现实的系统中，并不是所有的状态量都能直接获得，所以需要这样一个状态观测器来估计系统的状态量。

状态观测器的形式可以表示为：
$$
\hat{\mathbf{x}}(k+1) = \mathbf{F}\hat{\mathbf{x}}(k) + \mathbf{G}\mathbf{y}(k) + \mathbf{H}\mathbf{u}(k)
$$
<img src="https://notes.sjtu.edu.cn/uploads/upload_9df92dd309c6e5c15d61974477288d0e.png" style="zoom:80%;" />

观测器的设计准则：控制输入到真实状态的传递函数与到估计状态的传递函数一致（控制输入的作用一致）

$$
\begin{align}
\mathbf{x}(k + 1) = \mathbf{A}\mathbf{x}(k) + \mathbf{B}\mathbf{u}(k) \Rightarrow& \mathbf{X}(z) = (z\mathbf{I} - \mathbf{A})^{-1}\mathbf{BU}(z) \\
\hat{\mathbf{x}}(k+1) = \mathbf{F}\hat{\mathbf{x}}(k) + \mathbf{G}\mathbf{y}(k) + \mathbf{H}\mathbf{u}(k) \Rightarrow& \mathbf{\hat{X}}(z) = (z\mathbf{I} - \mathbf{F})^{-1}[\mathbf{GY}(z) + \mathbf{HU}(z)] \\
& \mathbf{\hat{X}}(z) = (z\mathbf{I} - \mathbf{F})^{-1}[\mathbf{GC}(z\mathbf{I} - \mathbf{A})^{-1}\mathbf{B} + \mathbf{H}]\mathbf{U}(z)
\end{align}
$$

$$
(z\mathbf{I} - \mathbf{A})^{-1}\mathbf{B} = (z\mathbf{I} - \mathbf{F})^{-1}[\mathbf{GC}(z\mathbf{I} - \mathbf{A})^{-1}\mathbf{B} + \mathbf{H}] \\
\Rightarrow (z\mathbf{I} - \mathbf{A})^{-1}\mathbf{B} = (z\mathbf{I} - \mathbf{F} - \mathbf{GC})^{-1} \mathbf{H}
$$

因此，只需 $\mathbf{F} = \mathbf{A-GC}$，$\mathbf{H} = \mathbf{B}$ 即可。由此，得到了 **Luenberger observer**：

$$
\begin{align*}
\hat{\mathbf{x}}(k+1) &= [\mathbf{A} - \mathbf{GC}]\hat{\mathbf{x}}(k) + \mathbf{G}\mathbf{y}(k) + \mathbf{B}\mathbf{u}(k) \\
&= \mathbf{A}\hat{\mathbf{x}}(k) + \mathbf{G}[\mathbf{y}(k) - \mathbf{C}\hat{\mathbf{x}}(k)] + \mathbf{B}\mathbf{u}(k)
\end{align*}
$$

可以看到，该观测器是基于观测误差 $\mathbf{y}(k) - \mathbf{C}\hat{\mathbf{x}}(k)$ 对状态估计进行修正的。

那么，如何设计 $\mathbf{G}$ 使得该观测器对于状态的估计能够收敛到真实值？换句话说，使得 $\mathbf{e_x}(k) = \mathbf{x}(k) - \mathbf{\hat{x}}(k)$ 收敛到 0

$$
\begin{align}
&\begin{cases}
\mathbf{x}(k + 1) = \mathbf{A}\mathbf{x}(k) + \mathbf{B}\mathbf{u}(k) \\
\hat{\mathbf{x}}(k+1) = \mathbf{A}\hat{\mathbf{x}}(k) + \mathbf{G}[\mathbf{y}(k) - \mathbf{C}\hat{\mathbf{x}}(k)] + \mathbf{B}\mathbf{u}(k)
\end{cases} \\
\Rightarrow & 
\mathbf{x}(k + 1) - \hat{\mathbf{x}}(k+1) = [\mathbf{A} - \mathbf{GC}][\mathbf{x}(k) - \hat{\mathbf{x}}(k)] \\
\Rightarrow & \mathbf{e_x}(k+1) = [\mathbf{A} - \mathbf{GC}]\mathbf{e_x}(k)
\end{align}
$$

因此，可以通过设计 $\mathbf{G}$ 来使得系统 $[\mathbf{A-GC}]$ 的特征值绝对值均小于 1，来实现观测误差的收敛（快速收敛）。

$\mathbf{G}$ 的设计方法和上一小节 $\mathbf{K}$ 的设计方法是一致的。

因此，**通过设计观测器来实现系统状态估计的前提条件是，系统是能观的！**



* **State feedback with a Luenberger observer**  

  通过 Luenberger observer 得到状态估计值，进一步实现状态反馈：
  
  $$
  \begin{align}
      \begin{cases}
      \mathbf{x}(k + 1) = \mathbf{A}\mathbf{x}(k) - \mathbf{BK}\mathbf{\hat{x}}(k) + \mathbf{BN}\mathbf{r}(k) \\
      \hat{\mathbf{x}}(k+1) = [\mathbf{A-GC-BK}]\hat{\mathbf{x}}(k) + \mathbf{GC}\mathbf{x}(k) + \mathbf{BN}\mathbf{r}(k)
      \end{cases} 
      \\
      \Rightarrow
      \begin{bmatrix}
      \mathbf{x}(k+1) \\
      \hat{\mathbf{x}}(k+1)
      \end{bmatrix}
      =
      \begin{bmatrix}
      \mathbf{A} & -\mathbf{BK} \\
      \mathbf{GC} & \mathbf{A} - \mathbf{GC} - \mathbf{BK}
      \end{bmatrix}
      \begin{bmatrix}
      \mathbf{x}(k) \\
      \hat{\mathbf{x}}(k)
      \end{bmatrix}
      +
      \begin{bmatrix}
      \mathbf{BN} \\
      \mathbf{BN}
      \end{bmatrix}
      \mathbf{r}(k)
  \end{align}
  $$
  <img src="https://notes.sjtu.edu.cn/uploads/upload_e96f41e1e3e4e293cef944ad220a94c5.png" style="zoom:80%;" />

  对于这样一个复合系统，我们同样希望系统的状态是收敛的（同时希望 $\mathbf{x}(k+1), \hat{\mathbf{x}}(k+1)$ 收敛），即对系统的特征值有要求：
  
  $$
  \begin{align}
  & \begin{vmatrix}
  z\mathbf{I} - \mathbf{A} & \mathbf{BK} \\
  -\mathbf{GC} & z\mathbf{I} - \mathbf{A} + \mathbf{GC} + \mathbf{BK}
  \end{vmatrix} = 0 \\
  \text{列变换}\Rightarrow
  & \begin{vmatrix}
  z\mathbf{I} - \mathbf{A} + \mathbf{BK} & \mathbf{BK} \\
  z\mathbf{I} - \mathbf{A} + \mathbf{BK} & z\mathbf{I} - \mathbf{A} + \mathbf{GC} + \mathbf{BK}
  \end{vmatrix} = 0 \\
  \text{行变换}\Rightarrow
  &\begin{vmatrix}
  z\mathbf{I} - \mathbf{A} + \mathbf{BK} & \mathbf{BK} \\
  0 & z\mathbf{I} - \mathbf{A} + \mathbf{GC}
  \end{vmatrix} = 0 \\
  \Rightarrow
  &\left| z\mathbf{I} - \mathbf{A} + \mathbf{BK} \right| \left| z\mathbf{I} - \mathbf{A} + \mathbf{GC} \right| = 0
  \end{align}
  $$
  
  可以看到系统稳定的要求可以拆解成 原系统稳定 和 状态估计稳定，即 $\mathbf{K}$ 和 $\mathbf{G}$ 的设计是解耦的。 
