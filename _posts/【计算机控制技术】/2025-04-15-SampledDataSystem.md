---
layout:       post
title:        "【计算机控制技术】- Sampled-Data System"
author:       "Orchid"
header-style: text
catalog:      true
hidden:       false
tags:
    - 控制
    - notes
---

在上一章中，我们已经学习了信号的采样和重构。那么是时候来**分析涉及信号采样与重构的系统**了。在自动控制原理中，我们所学习的系统往往只是处理连续信号量的，如果一个系统中涉及信号的采样与重构，那么会出现既有离散信号又有连续信号的情况，这种系统我们称为 Sampled-Data System。而这连续与离散信号是很难在一种域（s域、z域）中分析的。那么这一章将给出这类系统的分析方法。

## Block Diagram Analysis of Sampled-Data Systems

* **Open-Loop Sampled-Data Systems**

  首先考虑简单的，开环采样信号系统（通常使用 ZOH）。

  <img src="https://notes.sjtu.edu.cn/uploads/upload_11e4b0693d868cf6957b1c27521060fb.png" style="zoom:67%;" />

  第一反应是，能不能找到信号 $E(s)$ 和 $C(s)$ 之间的传递函数，如果能，那么该系统的分析也就变成了对于系统传递函数的分析。可以吗？关键在于 $E(s)$ 和 $E^{\star}(s)$ 之间是否存在传递函数，答案是无。因为从 $E(s)$ 到 $E^{\star}(s)$ 经过了一个采样器 $T$，而很显然，这个采样器的隐藏含义是，将信号从连续转换为离散，因此是不能用传递函数表示这一过程的。

  那么如何分析这个系统呢？既然无法找到 $E(s)$ 和 $C(s)$ 之间的传递函数，我们**可以将 $E^{\star}(s)$ 视作系统的输入**，尝试寻找 $E^{\star}(s)$ 和 $C(s)$ 之间的关系：
  
  $$
  C(s) = E^{\star}(s) G(s)
  $$
  
  下面给出一条**非常重要的公式：**
  
  $$
  \text{If} \ C(s) = E^{\star}(s) G(s), \quad \text{then} \ C^*(s) = E^{\star}(s) G^*(s)
  $$
  
  证明：
  
  $$
  \begin{align}
  \text{已知：}&E^{\star}(s) = \sum_{n=0}^{\infty} e(nT) \varepsilon^{-nTs} \\
  &\Rightarrow C^*(s) = \sum_{n=0}^{\infty} e(nT) \varepsilon^{-nTs}G(s) \\
  &\Rightarrow c(t) = \sum_{n=0}^{\infty} e(nT) g(t-nT) \\
  &\Rightarrow c(kT) = \sum_{n=0}^{\infty} e(nT) g(kT-nT), \quad (\text{when } k \le n, \ g(kT-nT) = 0)\\
  &\Rightarrow C^*(s) = \sum_{k=0}^{\infty} c(kT) \varepsilon^{-kTs} = \sum_{k=n}^{\infty} [\sum_{n=0}^{\infty} e(nT) g(kT-nT)] \varepsilon^{-kTs}, \quad \text{let k-n=m} \\
  &\Rightarrow C^*(s) = \sum_{m=0}^{\infty}\sum_{n=0}^{\infty} e(nT) g(mT) \varepsilon^{-(m+n)Ts} = \sum_{n=0}^{\infty} e(nT) \varepsilon^{-nTs}\sum_{m=0}^{\infty} g(mT) \varepsilon^{-mTs} \\
  &\Rightarrow C^*(s) = E^{\star}(s)G^*(s)
  \end{align}
  $$
  
  同时要注意一些**错误的变式**：
  
  $$
  \text{If} \ C(s) = E(s) G(s), \quad \text{then} \ C^*(s) = E^{\star}(s) G^*(s)
  $$
  
  **注意上述这个式子是错误变式！！！！正确的如下：**
  
  $$
  \text{If} \ C(s) = E(s) G(s), \quad \text{then} \ C^*(s) = \bar{EG}^*(s)=[E(s)G(s)]^*
  $$

* **Closed-Loop Sampled-Data Systems**

  接着可以讨论一下稍微复杂的闭环采样信号系统：

  <img src="https://notes.sjtu.edu.cn/uploads/upload_b43cfe9dc20a0e69b06b1d28db24fa09.png" style="zoom:67%;" />

  在开环采样信号系统中，我们的做法是，**将 $E^{\star}(s)$ 视作系统的输入**；实际上可以进一步规范化这个操作，**进入 $T$ 的信号视作系统的输出信号，$T$ 输出的信号视作系统的输入信号**。也就是说，现在系统有两个输入信号，即 $R(s)$ 和 $E^{\star}(s)$；系统有两个输出信号，即 $E(s)$ 和 $C(s)$。分析这个系统，只需列出从输入信号到输出信号的传递函数即可：
  
  $$
  \begin{cases}
  	C(s) = E^{\star}(s)G(s) \\
  	E(s) = R(s)-C(s)H(s) \\
  \end{cases}
  $$
  
  会发现上述式子中，输入信号和输出信号并未解耦，需要适当的变换：
  
  $$
  \begin{align}
  E(s) &= R(s) - E^{\star}(s)G(s)H(s) \\
  E^{\star}(s) &= R^*(s) - E^{\star}(s)[G(s)H(s)]^* \\
  \Rightarrow E^{\star}(s) &= \frac{R^*(s)}{1 + [G(s)H(s)]^*} \\
  \Rightarrow C(s) &= \frac{R^*(s)G(s)}{1 + [G(s)H(s)]^*}, \quad C^*(s) = \frac{R^*(s)G^*(s)}{1 + [G(s)H(s)]^*}
  \end{align}
  $$

## Sampled Signal Flow Graph

既然已经很好的分析了系统框图，为什么还需要分析信号流图？因为，**当系统中存在多个采样器和环**，单纯的系统框图分析将会变得困难，需要使用信号流图来分析（可以使用**梅森公式**）。

**Example：**

<img src="https://notes.sjtu.edu.cn/uploads/upload_5e3e567d4a3c95a9bdcf81dcc325f355.png" style="zoom:80%;" />

在上述系统框图中，有两个采样器 $T$，首先遵循**进入 $T$ 的信号视作系统的输出信号，$T$ 输出的信号视作系统的输入信号**这一原则，然后**将系统框图化为信号流图**：

<img src="https://notes.sjtu.edu.cn/uploads/upload_52659c159ad2b08eb90c8eacae2e4002.png" style="zoom:80%;" />

接着，根据因果关系，列出 Cause-and-effect equations（输入输出方程）：

$$
E_1 = R - G_2 E_2^*, \ E_2 = G_1 E_1^* - G_2 H E_2^*, \ C = G_2 E_2^*
$$

然后，等式两边做 starred transform：

$$
E_1^* = R^* - G_2^* E_2^*, \quad E_2^* = G_1^* E_1^* - \overline{G_2 H}^* E_2^*, \quad C^* = G_2^* E_2^* 
$$

然后根据 starred transform 后的等式，重新绘制 Sampled signal flow graph：

<img src="https://notes.sjtu.edu.cn/uploads/upload_31ae9cdbeb97d9c74a28a47456d5f404.png" style="zoom:80%;" />

最后，根据梅森公式，求解系统真实输入和真实输出之间的关系:

$$
C^* = \frac{G_1^* G_2^*}{1 + G_1^* G_2^* + \overline{G_2 H}^*} R^*, \quad
C(s) = \frac{G_2(s) G_1^*(s)}{1 + G_1^*(s) G_2^*(s) + \overline{G_2 H}^*(s)} R^*(s)
$$

## **Mason’s Rule**

* **Mason’s Gain Formula:** $G = \frac{1}{\Delta}\sum_{k=1}^NP_k\Delta_k$

  * G = input-output transfer function

  * N = nums of forward paths

  * Pk = path gain of the kth forward path

  * Δ = determinant of the graph

    <img src="https://notes.sjtu.edu.cn/uploads/upload_bfbae49cbf79a845c1df0e6e328cd291.png" style="zoom:67%;" />

  * Δk = cofactor of the kth forward path

    *  the determinant with the loops touching the kth forward path removed
    *  从 $\Delta$ 可以得到 $\Delta_k$。观察第 k 条前向通路，和哪几个回路touching，将touching回路在 $\Delta$ 中对应的参数置零，就可得到 $\Delta_k$

* **General Case:** 

  * used to **compute the linear dependence** between the independent variable **xi** and a dependent variable **xj**.
    * **xi**：**source node**, and often called the input variable
    * **xj**
      * xj is a **sink node**
      * xj is a **mixed node**, add a branch with gain 1 to create a sink node
    * you cannot use Mason rule directly when xi is not a source node

* **How to use Mason’s Gain Formula**

  <img src="https://notes.sjtu.edu.cn/uploads/upload_36ecf8d87f64a940ac8594bcd389f9dd.png" style="zoom:67%;" />

  1. 确定回路以及其回路增益 L
     * $L_1 = -G_4H_1$
     * $L_2 = -G_2G_7H_2$
     * $L_3 = -G_2G_3G_4G_5H_2$
     * $L_4 = -G_6G_4G_5H_2$ 

  2. 确定 non-touching 回路组合
     * $L_1, L_2$

  3. 确定 $\Delta$
     * $\Delta = 1 - (L_1 + L_2 + L_3 + L_4) + (L_1L_2)$

  4. 确定前向通路，以及对应的$\Delta_k$
     * $P_1 = G_1G_2G_3G_4G_5, \Delta_1 = 1$
     * $P_2 = G_1G_2G_7, \Delta_2 = 1-L_1$
     * $P_3 = G_1G_6G_4G_5, \Delta_3 = 1$

  5. 代入公式

