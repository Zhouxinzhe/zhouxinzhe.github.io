---
layout:       post
title:        "【计算机控制技术】- Sampling and Reconstruction"
author:       "Orchid"
header-style: text
catalog:      true
hidden:       false
tags:
    - 控制
    - notes
---

## A/D and D/A Conversion

在计算机控制系统中，Digital computer 接收的信号必须是数字信号，而其输出信号也必须转化为模拟信号才能作用于物理世界的执行器和被控对象。因此必须建立数字信号和模拟信号的桥梁，A/D转换和D/A转换应运而生。

<img src="https://notes.sjtu.edu.cn/uploads/upload_a2ba9228f3fe4a73897c1babcfb16182.png" style="zoom:67%;" />

### A/D 转换

A/D转换一般需要三个步骤：采样、量化、编码。

<img src="https://notes.sjtu.edu.cn/uploads/upload_020b6a21f970b74012b790a871c658fc.png" style="zoom:67%;" />

1. **采样**

   通常来说，采样是将连续信号采样为离散信号。同时，在每一个采样周期内，对采样得到的信号需要进行**保持**（holding），为后续的量化和编码提供一个稳定的信号输入，直到下一个采样周期的到来，进行新一轮的采样和保持。

   所以采样通常需要一个 **sample-and-hold amplifier**：

   <img src="https://notes.sjtu.edu.cn/uploads/upload_7b44702c7a7e4844feb3efaa153c5048.png" style="zoom:50%;" />

   $V_i$ 是输入的连续信号，$L$ 是控制采样的信号，$V_o$ 是输出的采样信号。可以看到，因为电容和运放的存在，实现了采样信号的 holding。

2. **量化**

   经过采样，我们已经得到了时间上离散但是幅值依旧连续的信号，而量化这一步骤就是将幅值依旧连续的信号进一步转换为幅值离散的信号，也就是把幅值量程划分为几个台阶，这也是A/D转换过程中误差的主要来源。通常来说，有两种不同的量化操作：

   <img src="https://notes.sjtu.edu.cn/uploads/upload_63313a09371c6676622827e97a05115c.png" style="zoom: 67%;" />

   其中，Quantization step size $\Delta = \frac{full-scale \ range}{digital \ range} = \frac{FSR}{2^N}$，这里的 $N$ 表示后续编码的位数。

   通常来说，*Round-off 量化方式使用的更多，因为其最大量化误差为 $\frac{\Delta}{2}$，为 Truncation 的一半*。

3. **编码**

   这一过程，实际上是将量化后幅值离散的信号，用二进制编码表示，即量化后每一个幅值台阶都对应一个唯一的二进制编码。

   假设有 $L$ 个量化台阶，需要的编码位数 $N \ge log_2L$。

   当然，二进制编码方式很多，你可以从最小到大编码，即 $000,001,\cdots,111$；也可以用补码的方式编码，即 $100, 101, \cdots, 011$。

下面看一些具体的 A/D 转换器：

* **Successive-approximation A/D converter**

  Uses a binary search to determine the bits of the output number sequentially.

   <img src="https://notes.sjtu.edu.cn/uploads/upload_7d944f8093a8c60b262d364a3af4e210.png" style="zoom:67%;" />              <img src="https://notes.sjtu.edu.cn/uploads/upload_f34e769149612e9496a05aa501604251.png" style="zoom:50%;" />

  * 优势：
    * Low cost，所需的硬件相对耗费较少
    * 每个 cycle 都能确定一个 bit，因此 N 位编码只需 N 个 cycle
    * 可以处理相对较长的编码需求
  * 劣势：
    * 转换精度非常依赖于其中的D/A转换器
    * 需要一个 sample-and-hold amplifier 来保证转换期间采样信号的稳定

* **Dual-slope A/D converter**

  <img src="https://notes.sjtu.edu.cn/uploads/upload_8d0e376fc162b3c181596a7f0fee045a.png" style="zoom:80%;" />

  * 转换过程：
    * 将开关打到 $V_1$ ，输入的模拟信号给第一个运放的电容充电固定的时间 $n_1$ 个clocks
    * 将开关打到 $V_{ref}$，给电容放电，在放电的过程中，第二个运放始终输出高电平，使得 $S_2$ 开关闭合，此时 Clock 不断地给 Counter脉冲，Counter进行累计计数，直至电容放电结束，$S_2$ 开关断开。最终 Counter 中累计的脉冲数即为编码。
    * 基本原理是：幅值越大的模拟信号在固定时间内给电容的充电量越大，导致电容放电的时间越长，Counter计数越大
  * 优势：
    * Provides a very accurate result that does not depend on the exact values of the capacitor or the clock rate.
    * Integrates the input during step 1, which reduces noise and aliasing.
  * 劣势：
    * 速度相对较慢，N 位编码最长需要 $2^N$ 个周期来计数

*  **Flash A/D converter**

  <img src="https://notes.sjtu.edu.cn/uploads/upload_53f0d2798af2b2db0a6de84933bae81c.png" style="zoom:67%;" />

  转换原理非常简单，看懂电路即可。

  * 优势：
    * 非常快，一次性、一瞬间就好了
    * 不需要 Sample-and-hold circuit
  * 劣势：
    * 如果编码位数很长，则需要指数级的比较器和异或器，很耗费硬件
    * 精度受到串联电阻的精度的限制

### D/A转换

D/A转换需要将数字信号，也就是二进制编码，转换为时间连续的模拟信号。对应两个步骤，一是将二进制编码转换为具体幅值的信号，二是将时间上离散的幅值信号连续化，即 Decoder 和 Data Hold。

<img src="https://notes.sjtu.edu.cn/uploads/upload_c3d22fdfb80c1ea1cf006449ef58d63a.png" style="zoom:67%;" />

其中，Data Hold 和A/D中采样过程中所需的信号维持类似，通常有两种方式：ZOH（零阶保持器）、FOH（一阶保持器）

而 Decoder 则有不同的实现方式，下面将展示一些具体的 D/A 转换器（不同的 Decoder）：

* **Weighted adder D/A**

  <img src="https://notes.sjtu.edu.cn/uploads/upload_e9c8798ebddf56d9bce07fd196059e04.png" style="zoom:67%;" />

  简化模型为：

  <img src="https://notes.sjtu.edu.cn/uploads/upload_76ca5590ed1fd5fcd370e37e10ae0e5a.png" style="zoom:67%;" />

  其中：
  
  $$
  n = \sum_{i=1}^{N} B_i 2^{i-1}, \quad B_i = 0 \text{ or } 1 \\
  I = \frac{V_{\text{ref}}^+ - V_{\text{ref}}^-}{xR} + \frac{V_{\text{ref}}-}{R} \\
  V(n) = \frac{R_f}{R} \left[ V_{\text{ref}}^- + (V_{\text{ref}}^+ - V_{\text{ref}}^-) \sum_{i=1}^{N} \frac{B_i}{2^{N+1-i}} \right] = \frac{R_f}{R} (V_{\text{ref}}^- + n\Delta V) \\
  \Delta V = \frac{V_{\text{ref}}^+ - V_{\text{ref}}^-}{2^N}
  $$
  
  这种 D/A 转换器易于设计，但不可能精确地制造大范围的电阻器值相同的集成电路芯片（电阻范围太大了）

* **R-2R resistive-ladder D/A**

  <img src="https://notes.sjtu.edu.cn/uploads/upload_2b504a6511ebcac6056381a30f192d12.png" style="zoom:67%;" />

  这是*最常用的 D/A 转换器*。转换原理很简单，看懂电路即可（整体等效电阻为 R）。
  
  $$
  \frac{V_{\text{ref}}^+ - V_{\text{ref}}^-}{R} = 2^N I_1 \\
  I = \sum_{i=1}^{N} B_i 2^{i-1} I_1 + \frac{V_{\text{ref}}^-}{R} = nI_1 + \frac{V_{\text{ref}}^-}{R}\\
  V(n) = IR_f = \frac{R_f}{R} (V_{\text{ref}}^- + n\Delta V) \\
  \Delta V = I_1 R
  $$



## The Ideal Sampler

计算机控制系统中最常见的采样器是 finite-pulsewidth sampler，其采样时长设为 p（通常考虑等间隔采样，因为很容易扩展到其他）。

但是为了理论分析，需要对 finite-pulsewidth sampler 进行理想化，即假设**采样时长 p << 采样周期 T**，且**采样时长 p 足够短**，保证**采样时长内信号是常值**。由此得到了 Flat-top sampler:

<img src="https://notes.sjtu.edu.cn/uploads/upload_ac71c7b4fea597814a545f9504a7f65f.png" style="zoom:67%;" />

同时可以进一步理想化，即假设**忽略量化编码过程的转换时间和量化误差**，由此得到 **Ideal sampler**：

<img src="https://notes.sjtu.edu.cn/uploads/upload_22dd5f1cf2981826a0dc7fea72125d4d.png" style="zoom:67%;" />

可以对 Ideal sampler 及其输出进行数学上的表述：

$$
\delta_T(t) = \sum_{n=0}^{\infty} \delta(t - nT) = \delta(t) + \delta(t - T) + \cdots \\
e^*(t) = e(t) \delta_T(t) = e(t) \delta(t) + e(t) \delta(t - T) + \cdots = e(0) \delta(t) + e(T) \delta(t - T) + \cdots
$$

需要注意的是，**Ideal sampler 的输出 $e^*(t)$ 不是物理可实现的信号**，因为脉冲信号不是物理可实现的。



## Starred Transform

上一小节已经给出了原始信号和采样信号在时域中的关系，这一小节将讨论原始信号和采样信号在 s 域中的变换关系。

$$
\begin{align}
&e^*(t) = \sum_{n=0}^{\infty} e(nT) \delta(t - nT) \\
\Rightarrow \quad &E^*(s) = \sum_{n=0}^{\infty} e(nT) \varepsilon^{-nTs}
\end{align}
$$

其中，$e(t)$ 是采样器的输入信号，如果 $e(t)$ 在 $t = kT$ 时刻不连续，则 $e(kT) = e(kT^+)$ 。**$E^*(s)$ 被称为 $e(t)$ 的 Starred Transform**。

那么，除了 $E^*(s)$ 的定义，是否还有其它方法来计算 $E^*(s)$ 呢？毕竟求解无穷级数不是一件一直很简单的事情。

### 留数法

$$
E^*(s) = \sum_{\text{at poles of } E(\lambda)} \left[ \text{residues of } F(\lambda) \right], \text{ where } F(\lambda) = E(\lambda) \frac{1}{1 - \varepsilon^{-T(s - \lambda)}}
$$

学过数理方法的同学，应该知道什么是留数，当然不知道也没关系，下面给出求解留数的方法，掌握即可：

* $E(\lambda)$ 在 $\lambda=a$ 处有单极点，$F(\lambda)$ 在 $\lambda = a$ 处的留数为：
  
  $$
  (\text{residue})_{\lambda=a} = (\lambda-a)F(\lambda)|_{\lambda=a}
  $$

* $E(\lambda)$ 在 $\lambda=a$ 处有 m 重极点，$F(\lambda)$ 在 $\lambda = a$ 处的留数为：
  
  $$
  (\text{residue})_{\lambda=a} = \frac{1}{(m-1)!} \frac{d^{m-1}}{d\lambda} \left[ (\lambda - a)^m F(\lambda) \right] \bigg|_{\lambda=a}
  $$

**注意，当 $e(t)$ 中包含时延项时，留数法不再适用。**

### 泊松公式法

* 如果 $e(t)$ 在所有的采样时刻都是连续的
  
  $$
  E^*(s) = \frac{1}{T} \sum_{n=-\infty}^{\infty} E(s + j n \omega_s), \quad \omega_s = \frac{2\pi}{T}
  $$

* 如果 $e(t)$ 在部分采样时刻是不连续的，即存在跳变的
  
  $$
  E^*(s) = \frac{1}{T} \sum_{n=-\infty}^{\infty} E(s + j n \omega_s) + \frac{1}{2} \sum_{n=0}^{\infty} \Delta e(nT) \varepsilon^{-nTs}, \quad \omega_s = \frac{2\pi}{T}
  $$
  
  其中，$\Delta e(nT)$ 就是采样时刻的信号跳变量

* 特别的，如果 $e(t)$ 只在 $t=0$ 时刻是跳变的，其余采样时刻均连续
  
  $$
  E^*(s) = \frac{1}{T} \sum_{n=-\infty}^{\infty} E(s + j n \omega_s) + \frac{e(0^+)}{2}, \quad \omega_s = \frac{2\pi}{T}
  $$

### Properties of E*(s)

1. $E^*(s)$ 在虚轴方向上是周期变化的，周期为 $j\omega_s$

   从定义中很容易看出来

2. 如果 $E(s)$ 存在极点 $s=s_1$，则 $E^*(s)$ 存在极点 $s = s_1 + j m \omega_s, \quad m = 0, \pm 1, \pm 2, \ldots$

   从泊松公式中很容易看出来（零点不具备该性质）

<img src="https://notes.sjtu.edu.cn/uploads/upload_a07e977f87e8d3854113eefce2a12033.png" style="zoom:67%;" />



## Data Reconstruction

在 D/A 转换器那一小节，只讲了如何将二进制编码转化为对应的幅值信号，但其实还留下了一个疑问没有解决，如何将离散的幅值信号重新转化为时间连续信号，即 Holder 部分没有讲解。这一小节将介绍，如何基于离散的幅值信号实现信号的重建。

学过数字信号处理的同学应该都知道，香农定理：如果采样频率大于信号最高频的两倍，那么基于采样信号可以实现原信号的无损复原

<img src="https://notes.sjtu.edu.cn/uploads/upload_af1e5a7ddecb16d9a1716decc332427d.png" style="zoom:67%;" />

但这是有**前提条件**的，就是你**必须获取所有的离散信号后，才能实现信号复原**。但在实际应用过程中，你并不是全知的，即你对于离散信号的获取是逐步的，那么如何**在逐步获取离散信号的过程中实现实时的信号重构**，是这一小节要解决的问题。

### ZOH 零阶保持器

$$
e_n(t) = e(nT), \quad nT \le t \le(n+1)T
$$

​         <img src="https://notes.sjtu.edu.cn/uploads/upload_cfb7f4e1c4ceb65c8ee02a91e26a554c.png" style="zoom:67%;" />               <img src="https://notes.sjtu.edu.cn/uploads/upload_27d7232dc5c86f1b2ea7513f416f3286.png" style="zoom:80%;" />

零阶保持器的每一小段，可以看成两个阶跃响应相减。因此，ZOH的传递函数可以得到：

$$
G_{h0}(s) = \frac{E_o(s)}{E_i(s)} = \frac{1 - \varepsilon^{-Ts}}{s}
$$

可以对 ZOH 的频域特征（频域响应）进行分析：

$$
G_{h0}(j\omega) = \frac{2 \varepsilon^{-j(\omega T/2)}}{\omega} \left[ \frac{\varepsilon^{j(\omega T/2)} - \varepsilon^{-j(\omega T/2)}}{2j} \right] = T \frac{\sin(\omega T/2)}{\omega T/2} e^{-j(\omega T/2)} \\

\text{幅值响应：}|G_{h0}(j\omega)| = T \left| \frac{\sin(\pi \omega / \omega_s)}{\pi \omega / \omega_s} \right| \\

\text{相位响应：}\angle G_{h0}(j\omega) = -\frac{\pi \omega}{\omega_s} + \theta, 
\quad \theta = \begin{cases} 
0, & \sin(\pi \omega / \omega_s) > 0 \\
\pi, & \sin(\pi \omega / \omega_s) < 0 
\end{cases}
$$
<img src="https://notes.sjtu.edu.cn/uploads/upload_840312976a4c878831263895ba467cff.png" style="zoom:67%;" />

### FOH 一阶保持器

$$
e_n(t) = e(nT) + e'(nT)(t-nT), \quad nT \le t \le(n+1)T \\
e'(nT) = \frac{e(nT) - e[n(T-1)]}{T}
$$

 <img src="https://notes.sjtu.edu.cn/uploads/upload_cacc5ab0577b08483c3a5b21ccd3cb0f.png" style="zoom:80%;" /> <img src="https://notes.sjtu.edu.cn/uploads/upload_c0ba7b022ad16b8e93aecc973677765b.png" style="zoom:67%;" />

对于一阶保持器，每一个脉冲信号都会有两个周期的作用时间：

$$
e_o(t) = [1_{t \geq 0} - 1_{t \geq T} + \frac{1}{T} t 1_{t \geq 0}  - \frac{1}{T} t  1_{t \geq T}] 
- [\frac{1}{T} (t - T) 1_{t \geq T} - \frac{1}{T} (t - T) 1_{t \geq 2T}]
$$

因此，FOH的传递函数可以得到：

$$
G_{h1}(s) = \frac{E_o(s)}{E_i(s)} = \frac{1 + Ts}{T} \left[ \frac{1 - \varepsilon^{-Ts}}{s} \right]^2
$$

同样，可以对 FOH 的频域特征（频域响应）进行分析：

$$
\text{幅值响应：}|G_{h1}(j\omega)| = T \sqrt{1 + \frac{4\pi^2 \omega^2}{\omega_s^2} \left[ \frac{\sin(\pi \omega / \omega_s)}{\pi \omega / \omega_s} \right]^2} \\

\text{相位响应：}\angle G_{h1}(j\omega) = \tan^{-1} \left( \frac{2\pi \omega}{\omega_s} \right) - \frac{2\pi \omega}{\omega_s}
$$
<img src="https://notes.sjtu.edu.cn/uploads/upload_144bf28f3c88905458bd1705b25fcd18.png" style="zoom:67%;" />

### Comparison

* 在低频阶段，FOH的表现更好，一是幅值响应更适合低频滤波，二是相位滞后较小
* 但当频率增大后，ZOH的表现更好，相位滞后更小
* **实际应用中，ZOH是最常用的**，而*FOH因为是有记忆的，硬件上耗费更大*



<img src="https://notes.sjtu.edu.cn/uploads/upload_a6ff541f17583c7d59f26c9f35721568.png" style="zoom:80%;" />
