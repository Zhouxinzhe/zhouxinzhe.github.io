---
layout:       post
title:        "【计算机控制技术】- Z Transform Method"
author:       "Orchid"
header-style: text
catalog:      true
hidden:       false
tags:
    - 控制
    - notes
---

在上一章中，我们已经学习了如何分析涉及信号采样与重构的系统，即当一个系统中既有连续信号又有离散信号时，如何分析系统输入与系统输出之间的关系（求得*类传递函数*的关系）。但是不是够了呢？当然不是，在上一节课的分析全部是在 s 域中，因此分析结果往往包含 **$[\cdot]^*$ 项**，即 starred transform 项，这一项在系统分析中会有什么问题呢？

1. $E^*(s) = \sum_{n=0}^{\infty} e(nT) \varepsilon^{-nTs}$，在 starred transform 中存在非实数项 $\varepsilon^{-nTs}$，这一项往往在经过复杂运算后难以进行拉普拉斯反变换（可以通过时移性质来解决，但非实数项终究是不太容易处理）
1. $E^*(s) = \sum_{n=0}^{\infty} e(nT) \varepsilon^{-nTs}$ 包含无穷多个极点和零点，上一章也讲到，$E^*(s)$ 沿虚轴方向是周期的。因此，无法使用零极点分析的方法来分析系统的性质。

因此，我们需要一种更好的分析方法，来分析涉及信号采样与重构的系统。**可以采用 Z 变换**。

## Z Transform Method

* **starred transform 可以无痛转换到 Z transform**
  
  $$
  E^*(s) = \sum_{n=0}^{\infty} e(nT) \varepsilon^{-nTs} \\
  \text{let} \quad z =  \varepsilon^{Ts} \\
  E^*(z) = \sum_{n=0}^{\infty} e(nT) z^{-n}
  $$
  
  很轻松的转换成了 z 的无穷级数，并且是 rational function of z 。
  
  $$
  E(z) = E^*(s) \big|_{s = \frac{\ln z}{T}} \quad \text{or} \quad E^*(s) = E(z) \big|_{z = \varepsilon^{Ts}}
  $$

* **Example**

  <img src="https://notes.sjtu.edu.cn/uploads/upload_eb44ae7c95bea99596fd45095e26f098.png" style="zoom:67%;" />

  由上一章的分析方法，我们可以得到：
  
  $$
  C^*(s) = \frac{G^*(s)}{1 + G^*(s)} R^*(s)
  $$
  
  再根据上述的转换方法，可以得到 z 域下的关系式：
  
  $$
  C(z) = \frac{G(z)}{1 + G(z)} R(z)
  $$

* **Pulse transfer function 脉冲传递函数**

  我们将 z 域下的传递函数称为脉冲传递函数：
  
  $$
  G_{pulse}(z) \triangleq \frac{C(z)}{R(z)}
  $$
  
  **脉冲传递函数所表征的，是采样输入信号 $r(kT)$ 和输出信号在采样时刻的信号 $c(kT)$ 之间的关系。**注意它并不包含输出信号在采样时刻之间的任何信息。



我们已经将 starred transform 项转换成了 z transform。但是注意到，在 sampled-data system 中，我们并不是所有时候都是从 starred transform 向 z transform 变换的，我们有时候是**从一般的 s 域连续信号，向 z 域变换**的。那么对于这种情况我们该如何应对？

* **From $G(s)$ to $G(z)$**

  在 sampled-data system 中，$G(s)$ 一般可以表示为 $\left( \frac{1 - \varepsilon^{-sT}}{s} \right) G_p(s)$，即 ZOH 与 被控对象 相乘。一般而言，从 s 域转换为 z 域需要以下几个步骤：

  1. 拉氏反变换，得到时域信号 $g(t) = L^{-1}[\left( \frac{1 - \varepsilon^{-sT}}{s} \right) G_p(s)]$
  2. 对时域信号采样，得到采样信号 $g(kT), k = 0, 1, 2, \cdots$
  3. 对采样信号进行 z 变换，得到 z-transform 信号 $G(z) = Z[\{g(kT)\}] = \sum_{k=0}^\infty g(kT)z^{-k}$

  我们将上述这三个步骤，用符号 $Z[]$ 表示，即 $G(z) = Z[G(s)]$。注意该符号内含了这三个步骤。（**也有另一种理解，就是对 G(s) 做 starred transform 得到 G*(s)，然后无痛转到 G(z)，最后是一样的**。）
  
  $$
  G(z) = Z[\left( \frac{1 - \varepsilon^{-sT}}{s} \right) G_p(s)] = (1-z^{-1})Z[\frac{G_p(s)}{s}]
  $$
  实际计算过程中，可以**直接读表**，不用真的去计算这三个步骤。

  ![](https://notes.sjtu.edu.cn/uploads/upload_a000e5efdd41ad69d5afa9d797dcf275.png)

* **Example**

  <img src="https://notes.sjtu.edu.cn/uploads/upload_b1a4bf416f777655bd282286d42e618f.png" style="zoom:80%;" />

  可以建模为：

  <img src="https://notes.sjtu.edu.cn/uploads/upload_ab6f38a263f626843a138fe0ebc51ba5.png" style="zoom:80%;" />
  
  $$
  \begin{align}
  &U(z) = E(z)D(z) \Leftrightarrow U^*(s) = E^*(s)D^*(s) \\
  &C(s) = G(s)U^*(s) \Rightarrow C^*(s) = G^*(s)U^*(s) \\
  \Rightarrow &C(z) = Z[G(s)]U(Z) = (1-z^{-1})Z[\frac{G_p(s)}{s}]E(z)D(z)
  \end{align}
  $$



## Discrete-Time Systems

这一小节，将给出离散时间系统的不同表示方式（注意不是 sampled-data system，sampled-data system 需要变换到 z 域下才能称之为离散时间系统）

1. **Difference equation or transfer function**：
   
   $$
   m(k) + a_{n-1} m(k-1) + \cdots + a_0 m(k-n) = b_n e(k) + b_{n-1} e(k-1) + \cdots + b_0 e(k-n) 
   $$
   
   z 变换后可以得到脉冲传递函数：
   
   $$
   D(z) = \frac{M(z)}{E(z)} = \frac{b_n + b_{n-1} z^{-1} + \cdots + b_0 z^{-n}}{1 + a_{n-1} z^{-1} + \cdots + a_0 z^{-n}}
   $$
   
   考虑一般的脉冲传递函数：
   
   $$
   D(z) = \frac{b_n + b_{n-1} z^{-1} + \cdots + b_0 z^{-n}}{a_m + a_{m-1} z^{-1} + \cdots + a_0 z^{-m}}, \quad m, n \text{ are positive integers}
   $$
   
   需要考虑其**物理可实现性**，即输出信号不能依赖于未来时刻的输入信号。如果**脉冲传递函数中分母的最高次≥分子的最高次**，则称该传递函数为 $proper$；如果脉冲传递函数中分母的最高次严格大于分子的最高次，则称该传递函数为 $strictly \ proper$。只要脉冲传递函数是 $proper$，则其物理可实现。

2. **Simulation diagram**

   相当于框图法，同样有着三个基本单元：**加法器、乘法器、时延器**（连续时间系统中为积分器）。For example，
   
   $$
   m(k) = b_1 e(k) - b_0 e(k-1) - a_0 m(k-1)
   $$
   <img src="https://notes.sjtu.edu.cn/uploads/upload_f6cf2bfe96390af0fcd4193159d959ff.png" style="zoom:67%;" />

3. **Signal flow graph**

   <img src="https://notes.sjtu.edu.cn/uploads/upload_df821012015d3bdc34eb4fc0a66ce534.png" style="zoom:80%;" />

   信号流图的好处，就是可以使用**梅森公式**，系统化的求解离散时间系统的脉冲传递函数。

4. **State-variable method**

   建立状态空间模型，这一部分完全可以参照现代控制理论。包括最基本的如何从一般脉冲传递函数建立状态空间模型、脉冲传递函数串联或并联分解后如何建立状态空间模型，如何从状态空间模型反推出脉冲传递函数。完全可以类推。
