---
layout:       post
title:        "【计算机控制技术】- Direct Digital Controller Design"
author:       "Orchid"
header-style: text
catalog:      true
hidden:       false
tags:
    - 控制	
    - notes
---

上一章介绍了离散状态空间模型下的数字控制器直接设计方法，这一章将要介绍离散时间系统（z 域）建模下的数字控制器直接设计方法。

* 离散域建模：z 域表达式
* 设计目标：设计控制器，使得系统稳定（系统输出、控制输出0）、零稳态误差



## Discretization of Continuous-time System

$$
G(s) = \left(\frac{1 - e^{-sT}}{s}\right) G_p(s) \quad \Rightarrow \quad G(z) = (1 - z^{-1}) \mathcal{Z} \left[ \frac{G_p(s)}{s} \right]
$$

将连续时间系统离散化为离散时间系统，是基于采样周期 $T$ 的。采样周期的选择很大程度上影响了离散时间系统的性质，进而影响了后续数字控制器的设计。那么，在**离散化**的过程中可能会有哪些不好的事情发生呢？

#### Non-minimum-phase zeros

离散化的过程往往会在离散时间系统中引入新的零点。而这些零点中可能会有非最小相位零点（在单位圆外），这会对数字控制器的设计有额外的要求和约束。接下来要解释，为什么离散化的过程会引入新的零点。

* 考虑阶跃输入，则连续时间系统和离散时间系统的阶跃响应分别为：
  
  $$
  C(s) = \frac{G_p(s)}{s}\\
  C(z) = \frac{zG(z)}{z-1}
  $$

* 由初值定理，可以得到 $\hat{c}(0)$ 和 $c(0)$:
  
  $$
  \hat{c}(0) = \lim_{s\rightarrow \infty} s\frac{G_p(s)}{s} = G_p(\infty) \\
  c(0) = \lim_{z\rightarrow\infty} \frac{zG(z)}{z-1} = G(\infty)
  $$

* 由于离散时间系统是由连续时间系统采样得到的，即 $\hat{c}(0) = c(0)$，则 $G_p(\infty) = G(\infty)$。因此，如果 $G_p(s)$ 是 $proper$，则 $G(z)$ 是 $proper$；如果 $G_p(s)$ 是 $strictly\ proper$，则 $G(z)$ 是 $strictly \ proper$。

  假设 $G(z)$ 是 $strictly \ proper$ 的：
  
  $$
  G(z) = \frac{b_1 z^3 + b_2 z^2 + b_3 z + b_4}{z^4 + a_1 z^3 + a_2 z^2 + a_3 z + a_4}
  $$
  
  则其阶跃响应可以表示为：
  
  $$
  C(z) = G(z) \frac{z}{z-1} = 0 + b_1 z^{-1} + [b_2 - b_1(a_1 - 1)] z^{-2} + \cdots \implies c(0) = 0, c(T) = b_1
  $$
  
  同样由于离散时间系统是由连续时间系统采样得到的，即 $\hat{c}(T) = c(T)$。而一般来说 $\hat{c}(T) \neq 0$，则 $c(T) = b_1 \neq 0$。

  也就是说，只要 $G_p(s)$ 是 $strictly\ proper$ 的，那么 $G(z)$ 分子的次数一定是比分母次数小 1 的，而 $G_p(s)$ 分子的次数比分母的次数可能小很多。这就是为什么 $G(z)$ 会比 $G_p(s)$ 有更多的零点。

#### Hidden Dynamics 

简单来说，就是离散化的过程把原本的动态特性给隐藏掉了。首先给一个具体的例子：

<img src="https://notes.sjtu.edu.cn/uploads/upload_c0f8ec13e7378f00a073efdaa9179527.png" style="zoom:80%;" />

右图的离散化过程中，恰巧所有的采样点都在震荡的最低点。因此，采样信号是完全不体现原本连续信号的震荡特性的。那么，为什么连续信号原本的动态特性可能在采样的过程中消失呢？首先，**系统的动态特性和系统的极点相关**。在离散化的过程中，系统的极点发生了什么？从 s 平面映射到了 z 平面。是一对一映射吗？不是，是压缩映射。这意味着 s 平面中的两个不同极点映射到 z 平面中成了同一个位置的极点。如下图，s 域中原本包含虚部的极点映射到 z 域中成了实轴上的极点，很明显原本的动态震荡就消失了。

<img src="https://notes.sjtu.edu.cn/uploads/upload_d1ca7a6435525672b36e042fadd1d673.png" style="zoom:80%;" />

#### Loss of Controllability and Observability  

这一点是从状态空间模型的角度出发的。至于为什么可能会失去能观性和能控性，这一点尚在研究，没有很好的相关解释。但是有前人给出了一个充分条件，来保证不失去原本的能观性或能控性：

$$
\begin{align}
&\mathbf{x}(k+1) = \mathbf{A}\mathbf{x}(k) + \mathbf{B}\mathbf{u}(k)\\
&\mathbf{y}(k) = \mathbf{C}\mathbf{x}(k)\\
&\mathbf{A} = \varepsilon^{\mathbf{A}_a T}, \quad \mathbf{B} = \int_0^T \varepsilon^{\mathbf{A}_a \tau} d\tau \mathbf{B}_a, \quad \mathbf{C} = \mathbf{C}_a
\end{align}
$$

A sufficient condition for its discretized state-variable equation with sampling period $T$ to be controllable (or observable) is that

$$
|\text{Im}(\lambda_i - \lambda_j)| \neq \frac{2m\pi}{T}, \, \forall m = 1, 2, \ldots, \text{ whenever } \text{Re}(\lambda_i - \lambda_j) = 0
$$


## Time-Domain Design  

* z 域中，一般的数字控制器表达式如下：

    $$
    D(z) = \frac{U(z)}{E(z)} = \frac{a_0 + a_1 z^{-1} + \cdots + a_m z^{-m}}{b_0 + b_1 z^{-1} + \cdots + b_l z^{-l}}
    $$

    由此，可以得到时域的表达式：
    
    $$
    b_0 u(k) + b_1 u(k-1) + b_2 u(k-2) + \cdots + b_l u(k-l) = a_0 e(k) + a_1 e(k-1) + \cdots + a_m e(k-m) \\
    \implies u(k) = \frac{1}{b_0} \left[ \sum_{i=0}^{m} a_i e(k-i) - \sum_{j=1}^{l} b_j u(k-j) \right]
    $$
    
    这种表达式要求储存 $O(m+l)$ 量级的数据，可以进行适当的变换：
    
    $$
    F(z) = \frac{E(z)}{\sum_{j=0}^{l} b_j z^{-j}} \implies  U(z) = F(z) \sum_{i=0}^{m} a_i z^{-i} \\
    \implies \quad \left\{
    \begin{aligned}
    f_k &= \frac{1}{b_0} \left[ e(k) - \sum_{i=1}^{l} b_i f_{k-i} \right] \\
    u(k) &= \sum_{i=0}^{m} a_i f_{k-i}
    \end{aligned}
    \right.
    $$
    
    这样仅需储存 $O(max\{m,l\})$ 量级的数据。

* 接着需要确定 $a_i,b_j$ 参数，可以采用优化的方法（感觉很唐）

    1. 采集 $M$ 步数据
       
       $$
       \{ e(0), e(1), \ldots, e(M) \}, \ \{ u(0), u(1), \ldots, u(M) \}
       $$

    2. 设置目标函数
       
       $$
       \min_{a_i, b_j} \ J = \sum_{k=0}^{M} \left[ e(k)^2 + \alpha (u(k) - u(\infty))^2 \right], \quad \alpha > 0 \text{ is a weight}
       $$

    3. 使用 convex optimization method   

* 考虑特殊输入：阶跃输入；考虑特殊目标：零稳态误差（保证系统稳定的前提）
    
    $$
    e(\infty) = \lim_{z \to 1} (z - 1) E(z) = \frac{1}{1 + \lim_{z \to 1} D(z) G(z)}
    $$
    
    为保证零稳态误差，只需 $D(1) = \infty$，即
    
    $$
    \sum_{j=0}^{l} b_j = 0
    $$
    
    考虑最简单的情况，$b_0 = 0, \quad b_1 = -1$
    
    $$
    D(z) = \frac{U(z)}{E(z)} = \frac{a_0 + a_1 z^{-1} + \cdots + a_m z^{-m}}{1 - z^{-1}}
    $$

    * $m = 1$
      
      $$
       D(z) = \frac{a_0 + a_1 z^{-1}}{1 - z^{-1}} = -a_1 + (a_0 + a_1) \frac{z}{z - 1}  \\
       \implies u(k) = u(k-1) + a_0 e(k) + a_1 e(k-1) \\
       \implies u(k) = a_0 e(k) + (a_0 + a_1) \sum_{j=0}^{k-1} e(j)
      $$

    * $m = 2$
      
      $$
      D(z) = \frac{a_0 + a_1 z^{-1} + a_2 z^{-2}}{1 - z^{-1}} = -(a_1 + 2a_2) + (a_0 + a_1 + a_2) \frac{z}{z - 1} + a_2 \frac{z - 1}{z} \\
      \implies u(k) = u(k-1) + a_0 e(k) + a_1 e(k-1) + a_2 e(k-2) \\
      \implies u(k) = a_0 e(k) + (a_0 + a_1 + a_2) \sum_{j=0}^{k-1} e(j) - a_2 e(k-1)
      $$



## Direct Synthesis  

上一节是从数字控制器的输入与输出的时域关系出发，进而得到数字控制器的表达式。这一节将从系统闭环传递函数的全局视角出发，反推得到期望的数字控制器。接下来将从两点进行解释：一，在得到期望的闭环传递函数后，如何反推相应的数字控制器；二，如何得到或者说如何选择期望的闭环传递函数。（逻辑上是先“二”后“一”，但是这里“二”是主体，所有放后面讲）

* **如何反推相应的数字控制器**
  
  $$
  G_{cl}(z) = \frac{C(z)}{R(z)} = \frac{D(z)G(z)}{1 + D(z)G(z)} \\
  \implies D(z) = \frac{1}{G(z)} \frac{G_{cl}(z)}{1 - G_{cl}(z)}
  $$
  
* **如何选择期望的闭环传递函数**

  事实上从不同的考虑角度出发，会有很多不同闭环传递函数选择方法。以下是一些方法的考虑角度：

  * **Dead-beat design**：在有限时间内达到零稳态误差
  * **Dahlin’s method**：得到不那么激进的闭环传递函数
  * **Internal model control**：克服系统建模的误差（uncertainty）和环境干扰（disturbance）

#### Dead-Beat Design

在连续时间系统中，我们可以设计系统的闭环传递函数，使得在时间趋于无穷时，系统的稳态误差为零（针对特定的输入）；那么在离散时间系统中，我们能否设计系统的闭环传递函数，**使得在有限时间内，系统的稳态误差就达到零呢**？（同样是针对特定输入）

* **unit-step input 阶跃输入**

  * Problem
    
    $$
    \text{Given } r(k) = 1, \forall k \geq 0, \text{ find } G_{cl}(z) \text{ such that } c(k) = 1, \forall k \geq N \text{ for some } N \geq 0
    $$

  * Answer

    根据要求可以列出系统的输入和响应：
    
    $$
    \begin{align}
    &C(z) = c_0 + c_1 z^{-1} \cdots + c_{N-1} z^{-N+1} + \sum_{k=N}^{\infty} z^{-k}\\
    &R(z) = \frac{z}{z-1}
    \end{align}
    $$
    
    其中，$c_0, c_1, \cdots, c_{N-1}, N$ 均是待定的参数，可以任意选取。基于此可以给出闭环传递函数的表达式：
    
    $$
    \begin{align}
    G_{cl}(z) &= \frac{c_0 + c_1 z^{-1} \cdots + c_{N-1} z^{-N+1} + \sum_{k=N}^{\infty} z^{-k}}{\frac{z}{z-1}}\\
    &= \frac{\frac{z}{z-1} + [c_0 - 1 + (c_1 - 1) z^{-1} + \cdots + (c_{N-1} - 1) z^{-N+1}]}{\frac{z}{z-1}}\\ 
    &= c'_0 + c'_1 z^{-1} + \cdots + c'_{N} z^{-N}
    \end{align}
    $$
    
    其中，$c'_0, c'_1, \cdots, c'_{N-1}$ 可以由 $c_0, c_1, \cdots, c_{N-1}$ 表示，并非新的参数。

* **unit-ramp  input 斜坡输入**

  * Problem
    
    $$
    \text{Given } r(k) = kT, \forall k \geq 0, \text{ find } G_{cl}(z) \text{ such that } c(k) = kT, \forall k \geq N \text{ for some } N \geq 0
    $$

  * Answer
    
    根据要求可以列出系统的输入和响应：
    
    $$
    \begin{align}
    &C(z) = c_0 + c_1 z^{-1} \cdots + c_{N-1} z^{-N+1} + \sum_{k=N}^{\infty} kT z^{-k}\\
    &R(z) = \frac{Tz}{(z-1)^2}
    \end{align}
    $$
    
    其中，$c_0, c_1, \cdots, c_{N-1}, N$ 均是待定的参数，可以任意选取。基于此可以给出闭环传递函数的表达式：
    
    $$
    \begin{align}
    G_{cl}(z) &= \frac{c_0 + c_1 z^{-1} \cdots + c_{N-1} z^{-N+1} + \sum_{k=N}^{\infty} kT z^{-k}}{\frac{Tz}{(z-1)^2}}\\
    &= \frac{\frac{Tz}{(z-1)^2} + [c_0 + (c_1 - T) z^{-1} + \cdots + (c_{N-1} - (N-1)T) z^{-N+1}]}{\frac{Tz}{(z-1)^2}}\\ 
    &= c'_0 + c'_1 z^{-1} + \cdots + c'_{N} z^{-N}
    \end{align}
    $$
    
    其中，$c'_0, c'_1, \cdots, c'_{N-1}$ 可以由 $c_0, c_1, \cdots, c_{N-1}$ 表示，并非新的参数。

* unit-acceleration input  留作读者的练习

可以看到，对于特定的系统输入（unit-step、unit-ramp and unit-acceleration inputs），在期望有限时间内消除稳态误差的前提下设计得到的系统闭环传递函数有共同的表达式：

$$
G_{cl}(z) = \frac{c_0 z^N + c_1 z^{N-1} + \cdots + c_N}{z^N} = c_0 + c_1 z^{-1} + c_2 z^{-2} + \cdots + c_N z^{-N}, N \geq 1 \tag{1}
$$

其特征就是**将系统的所有极点都配置到 $z = 0$ 处**。通过 s 平面和 z 平面的映射关系 $z = \varepsilon^{sT}$ 可以得到，$z=0$ 对应的是 $s = -\infty$，因此在连续时间系统无法使用 Dead-Beat Design 的设计思路使得系统在有限时间内消除稳态误差。（dead-beat design is unique to digital systems）

* **设计须考虑的约束——物理可实现**

  上述所设计出来的闭环传递函数是不加约束的，简单来说就是并不能保证反推得到的数字控制器 $D(z)$ 是**物理可实现**的。那要如何考虑来保证 $D(z)$ 的物理可实现性呢？

  * $D(z)$ 物理可实现：分子最高次小于等于分母最高次（最高次幂小于等于0）

  * 如果 $G_p(s)$ 包含延时项 $\varepsilon^{-Ls}, L > 0$，那么设计出来的闭环传递函数必须至少包含 $z^{-l}, L=lT$ 的延时项
    
    $$
    G(s) \varepsilon^{-Ls} \longrightarrow g(t - L) 1_{t \geq L} \longrightarrow g(kT - L) 1_{k \geq L/T} \longrightarrow z^{-l} G(z) \quad \text{ assume } L = lT, l \text{ is an integer}
    $$

  * 当 $G(z)$ 展开成 $z^{-1}$ 的多项式，$G_{cl}(z)$ 的最高次幂需要小于等于 $G(z)$ 的最高次幂（$D(z)$ 的最高次幂小于等于0）。所以当 $G(z)$ 的最高次项为 $-1$ 时，$G_{cl}(z)$ 的最高次项为最大为 $-1$，即
    
    $$
    G_{cl}(z) = c_1 z^{-1} + c_2 z^{-2} + \cdots + c_N z^{-N}
    $$

  * 对于大部分的 plant，都会包含延时项；特别是离散时间系统，$G(z)$ 往往包含延时项（零时刻给的输入不会在零时刻的输出直接反应），因此
    
    $$
    G_{cl}(z) = c_1 z^{-1} + c_2 z^{-2} + \cdots + c_N z^{-N}, N \geq 1 \tag{2}
    $$
    
    其中，$c_0, c_1, \cdots, c_{N-1}, N$ 可以任意选取。



我们进一步给出了考虑物理可实现约束的闭环传递函数的一般形式，那么能不能给出具体的解呢？同样我们考虑特定的输入，unit-step、unit-ramp and unit-acceleration inputs，给出其闭环传递函数的具体可行解。

* 将三种特定输入统一描述：
  
  $$
  R(z) = \frac{A(z)}{(1-z^{-1})^q}
  $$

  * unit-step：$1_{t\geq0}, \quad \frac{1}{1-z^{-1}}, \quad q = 1$
  * unit-ramp：$t, \quad \frac{Tz^{-1}}{(1-z^{-1})^2}, \quad q=2$
  * unit-acceleration：$\frac{t^2}{2}, \quad \frac{T^2z^{-1}(1+z^{-1})}{2(1-z^{-1})^3}, \quad q =3$

  接着**考虑系统的稳态误差**：
  
  $$
  \begin{align}
  &E(z) = [1 - G_{cl}(z)] R(z) = \frac{A(z) [1 - G_{cl}(z)]}{(1 - z^{-1})^q} \\
  &e(\infty) = \lim_{z\rightarrow1}(z-1)E(z) = (z-1)\frac{A(z) [1 - G_{cl}(z)]}{(1 - z^{-1})^q} \tag{3}
  \end{align}
  $$
  
  因为 $A(1) \neq 0$，为了保证 $e(\infty) = 0$，则 $[1-G_{cl}(z)]$ 中必须包含 $(1 - z^{-1})^q$ ，即
  
  $$
  1 - G_{cl}(z) = (1 - z^{-1})^q F(z) \\
  F(z) = f_0 + f_1 z^{-1} + \cdots + f_p z^{-p}, \quad p = N - q
  $$
  
  由 $(2)$ 式可知，$G_{cl}(z)$ 的常数项 $c_0 = 0$，因此 $f_0 = 1$，则有：
  
  $$
  G_{cl}(z) = 1 - (1 - z^{-1})^q (1 + f_1 z^{-1} + \cdots + f_p z^{-p}) \tag{4}
  $$

至此，进一步得到了遵循Dead-Beat Design 的设计思路下，针对特定的输入的闭环传递函数的表达式 $(4)$，但这还不够，还有不定的参数 $f_1, f_2, \cdots, f_p, p$，这些参数仍是能够自由选取的。最简单的，令 $f_1, f_2, \cdots, f_p, p = 0$，由此可以得到 Dead-Beat Design 设计中的 **minimal prototype controller** 。

* **minimal prototype controller**
  
  $$
  G_{cl}(z) = 1 - (1 - z^{-1})^q \tag{5}
  $$

  * $q = 1, \quad G_{cl}(z) = z^{-1}, \quad D(z) = \frac{1}{G(z)}\frac{z^{-1}}{1-z^{-1}}, \quad C(z) = \frac{z^{-1}}{1-z^{-1}}$
  * $q = 2, \quad G_{cl}(z) = 2z^{-1}-z^{-2}, \quad D(z) = \frac{1}{G(z)}\frac{2z^{-1}-z^{-2}}{1-2z^{-1}+z^{-2}}, \quad C(z) = \frac{Tz^{-1}(2z^{-1}-z^{-2})}{(1-z^{-1})^2}$
  * $q = 2, \quad G_{cl}(z) = 3z^{-1} - 3z^{-2} + z^{-3}, \quad D(z) = \frac{1}{G(z)}\frac{3z^{-1} - 3z^{-2} + z^{-3}}{1-3z^{-1} + 3z^{-2} - z^{-3}}, \quad C(z) = \frac{T^2z^{-1}(1+z^{-1})}{2(1-z^{-1})^3}(3z^{-1} - 3z^{-2} + z^{-3})$

  需要注意的是，Minimal prototype controllers 是针对这三种特定的输入精简得到的，不保证对于其他输入的泛化性。



到这就结束了吗？是不是忘记了什么？没错，**还需要考虑系统的稳定性（stability）！**这是最最最重要的！我们遵循Dead-Beat Design 的设计思路设计得到的闭环传递函数能够保证系统输出的稳定性（因为闭环传递函数的极点都在 0），但是并不能够保证系统内部信号的稳定性。为什么要考虑系统内部？又不是状态空间模型？**因为我们最终的设计目标是数字控制器，要保证数字控制器的输出是稳定的**。

<img src="https://notes.sjtu.edu.cn/uploads/upload_265ce8aa592b1d0b20d15c75965082c0.png" style="zoom:80%;" />

* **数字控制器输出稳定 Stability**
  
  $$
  U(z) = \frac{C(z)}{G(z)} = \frac{G_{cl}(z)R(z)}{G(z)}
  $$
  
  考虑一个信号的稳定性，则看该信号的极点。要保证控制器的输出稳定，则要求 $U(z)$ 的极点都是在单位圆内。而 $U(z)$ 中可能提供单位圆外的极点的只有 $\frac{1}{G(z)}$ 项。这里就要回到本章第一节提到的离散化过程中可能出现的问题“引入非最小相位零点”，也就是说 $G(z)$ 可能包含单位圆外的零点，这就导致 $U(z)$ 存在单位圆外的零点，进而导致其输出不稳定。那么该如何修正呢？很简单，**改变我们设计的期望闭环传递函数来抵消单位圆外的极点即可**：
  
  $$
  \text{Let } |z_k| \geq 1, k = 1, 2, \ldots, v \text{ be non-minimum-phase zeros, then } G_{cl}(z) \text{ should be modified as follows} \\
  \begin{align}
  &\text{Stability:} \quad G_{cl}(z) = (1 - z_1 z^{-1}) \cdots (1 - z_v z^{-1}) G'_{cl}(z)\\
  &\text{Zero steady-state error:} \quad 1 - G_{cl}(z) = (1 - z^{-1})^q F(z)
  \end{align}
  $$
  
  这时候，我们就不能简单的令 $F(z) = 1$ 了。



到这还没结束，还要考虑一些实际的问题。

* **Model Mismatch**

  首先，直接设计法的基本前提是已知系统建模 $G(s)$ ，通过设计期望的闭环传递函数 $G_{cl}(s)$ 从而反推得到数字控制器 $D(z)$ 。在实际情况中，我们对于系统的建模 $\hat{G}(s)$ 与系统的真实模型 $G(s)$ 是存在细微的误差的。这种误差会有什么影响？可以看一个具体的例子：
  
  $$
  G(z) = \frac{2.2}{z + \alpha}, \quad \hat{G}(z) = \frac{2.2}{z + \hat{\alpha}}, \quad \text{unit-step input} \\
  D(z) = \frac{1}{\hat{G}(z)} \frac{\hat{G}_{cl}(z)}{1 - \hat{G}_{cl}(z)} = \frac{1 + \hat{\alpha} z^{-1}}{2.2(1 - z^{-1})} = \frac{z + \hat{\alpha}}{2.2(z - 1)}, \quad \hat{G}_{cl}(z) = z^{-1} \\
  1 + D(z)G(z) = 0 \implies z^2 + \alpha z + \hat{\alpha} - \alpha = 0
  $$
  
  接着用 Jury 判据可以得到关于 $\alpha$ 和 $\hat{\alpha}$ 的稳定域：
  
  $$
  \begin{align*}
  Q(1) > 0 &\implies \hat{\alpha} > -1 \\
  (-1)^2 Q(-1) > 0 &\implies \hat{\alpha} > 2\alpha - 1 \\
  |a_0| < a_2 &\implies \alpha - 1 < \hat{\alpha} < \ < \alpha + 1
  \end{align*}
  $$
  
  <img src="https://notes.sjtu.edu.cn/uploads/upload_c5c064e91c91034b28c538b00a37abef.png" style="zoom: 50%;" />

  会发现非常不合理的点是，并不是所有的 $\alpha$ 只要我 $\hat{\alpha}$ 估计的够准，得到的系统就一定是稳定的。那么该怎么解决这个问题呢？老师给出了一种方法（虽然我不是很理解为什么）：**用 $1 - \hat{G}_{cl}(z)$ 来消去 $\hat{G}(z)$ 的分母**
  
  $$
  1 - \hat{G}_{cl}(z) = (1 + \hat{\alpha} z^{-1})(1 - z^{-1}) F(z)
  $$

  $$
  D(z) = \frac{1}{\hat{G}(z)} \frac{\hat{G}_{cl}(z)}{1 - \hat{G}_{cl}(z)} = \frac{(1 - \hat{\alpha} + \hat{\alpha} z^{-1}}{2.2(1 - z^{-1})} \\
  1 + D(z)G(z) = 1 + (\alpha - \hat{\alpha})z^{-1} + (\hat{\alpha} - \alpha)z^{-2} = 0 \implies z^2 + (\alpha - \hat{\alpha})z + \hat{\alpha} - \alpha = 0 \implies \alpha - 0.5 < \hat{\alpha} < \alpha + 1
  $$

  <img src="https://notes.sjtu.edu.cn/uploads/upload_2ad1c07741fd5684ee9420a16d344fcc.png" style="zoom:50%;" />

  通过上述方法设计得到的闭环传递函数 $\hat{G}_{cl}(z)$ ，只要我 $\hat{\alpha}$ 估计的够准，系统必定是稳定的。



* **Intersample ripple**

  什么是采样间波纹呢？首先，我必须重新再提一嘴，我们真正考虑的系统是计算机的 **Sampled Data System**，而为了更简便的考虑，我们才将其转换为了 z 域下的离散时间系统。而采样信号系统的输出实际上是连续信号 $C(s)$，而我们简化后的离散时间系统的输出实际上是对 $C(s)$ 的采样得到的 $C(z)$。
  
  我们通过上述 Dead-Beat Design 所有的精巧设计，目的就是在有限时间内使得离散输出 $C(z)$ 跟上系统输入 $R(z)$。但有一个问题是，我们忽略了两个采样时刻之间的信号，他可能是上下波动的，而刚好只是在所有的采样时刻对上了系统输入，如下图所示：
  
  <img src="https://notes.sjtu.edu.cn/uploads/upload_e58e6b744cbb0bb50057cc06089ae345.png" style="zoom:50%;" />
  
  但我们实际上希望的是 $C(s)$ 跟上系统输入，也就是希望消除采样间波纹。如何消除？有一个非常强有力的保证，如果我能够保证**数字控制器在稳态时刻的输出控制量收敛到了一个常值**，那么采样信号系统中的真实控制量是离散控制量通过 ZOH 得到的也将是一个常量，很明显对于常量控制量，采样信号系统的输出也将不会波动（假设已经达到稳态），如此即能消除波纹（这是一个非常强的条件）。用数学语言描述一下：
  
  $$
  u(k) = 0, \text{ or } u(k) = c, \text{ } c \text{ is a constant, } \forall k \geq N 
  $$
  
  $$
  \begin{align}
  U(z) &= c_0 + c_1 z^{-1} \cdots + c_{N-1} z^{-N+1} + \sum_{k=N}^{\infty} z^{-k}\\
  &=  [c_0 - 1 + (c_1 - 1) z^{-1} + \cdots + (c_{N-1} - 1) z^{-N+1}] + \frac{1}{1-z^{-1}}\\ 
  &= [c'_0 + c'_1 z^{-1} + \cdots + c'_{N} z^{-N}]*\frac{1}{1-z^{-1}} \\
  U(z) &= \frac{G_{cl}(z)R(z)}{G(z)} = \frac{G_{cl}(z)}{N_G(z)} \frac{M_G(z)A(z)}{(1-z^{-1})^q}
  \end{align}
  $$
  
  因此，我们所期望的式子如下：
  
  $$
  \frac{G_{cl}(z)}{N_G(z)} \frac{M_G(z)A(z)}{(1-z^{-1})^q} = [c'_0 + c'_1 z^{-1} + \cdots + c'_{N} z^{-N}]*\frac{1}{1-z^{-1}}
  $$
  
  该如何使得上述式子成立呢？两个条件：
  
  * $G_{cl}(z)$ 包含 $G(z)$ 的所有零点，那么 $\frac{G_{cl}(z)}{N_G(z)}$ 就可以除尽
  * $G(z)$ 包含 $q-1$ 个 $z = 1$ 的极点，那么 $\frac{M_G(z)}{(1-z^{-1})^q}$ 就可以给出 $\frac{1}{1-z^{-1}}$
  
  第一个条件我们可以用于设计期望的闭环传递函数，第二个条件则是对系统的 Plant 提出了要求。



到这里，关于 Dead-Beat Design 所要考虑的事情大概都已经结束了。感觉脑子非常的混乱，因为考虑来太多的事情，接下来给出总结：

* **总结**

  * Deat-Beat Design 的基本点——所有极点在零点：
    
    $$
    G_{cl}(z) =  c_0 + c_1 z^{-1} + c_2 z^{-2} + \cdots + c_N z^{-N}
    $$

  * 考虑物理可实现性（往往包含时延项）：
    
    $$
    G_{cl}(z) =  c_1 z^{-1} + c_2 z^{-2} + \cdots + c_N z^{-N}
    $$

  * 考虑三种特定输入的零稳态误差：
    
    $$
    1 - G_{cl}(z) = (1 - z^{-1})^q F(z)
    $$

  * 考虑数字控制器输出稳定 Stability，$G_{cl}(z)$ 包含 $G(z)$ 非最小项零点：
    
    $$
    G_{cl}(z) = (1 - z_1 z^{-1}) \cdots (1 - z_v z^{-1}) G'_{cl}(z)
    $$

  * 考虑 Model Mismatch，$1 - G_{cl}(z)$ 包含 $G(z)$ 在单位圆上或外的极点：
    
    $$
    1 - G_{cl}(z) = (1 - p_1^+ z^{-1}) \ldots (1 - p_v^+ z^{-1})(1 - z^{-1})^q F(z)
    $$

  * 考虑消除 intersample ripple，$G_{cl}(z)$ 包含 $G(z)$ 所有零点：
    
    $$
    G_{cl}(z) = (1 - z_1 z^{-1}) \cdots (1 - z_v z^{-1})\cdots (1 - z_w z^{-1}) G'_{cl}(z)
    $$
    













