---
layout:       post
title:        "【计算机控制技术】- Digital Controller Design "
author:       "Orchid"
header-style: text
catalog:      true
hidden:       false
tags:
    - 控制	
    - notes
---

我们已经学习了**离散时间系统的建模**、**离散时间系统的分析**。因此紧接着需要学习的就是，**离散时间系统的控制设计**。通过设计离散域中的控制器，使得系统满足我们的要求。

但是 digital controller 应该如何设计呢？我们还是按照一贯的思路，先看看连续时间系统那一套。在自动控制原理中，我们已经学习了如何设计控制器（补偿矫正、PID）使得系统满足设计要求。那么到离散域中，我们需要重新再来一遍吗？注意，离散时间系统实际上也是从连续时间系统中离散化得到的，所以我们只需**针对离散时间系统对应的连续时间系统设计好模拟控制器，然后离散化成 digital controller**。而针对连续时间系统设计模拟控制器是自动控制原理解决的问题，这里不再赘述。因此这一章解决的问题是，**如何将 Analog Controller 离散化成 Digital Controller，并且保证两者在连续和离散域下的性质尽可能一致**。（不可能使得离散后的控制器的性质和原本的完全一致的，除非采样时间 T 趋于 0，我们要做的就是尽可能保证相似）



## Time-domain Invariance Method

首先可以从时域的角度来思考，怎样的 digital controller 和 analog controller 的性质会完全一致呢？

![](https://notes.sjtu.edu.cn/uploads/upload_e602e35335efe3e069c8d458ad80aaaa.png)

* 最天真的想法：对于任意同样的输入 $e(t)$，上述两个控制器的输出 $\hat{u}(t) = \bar{u}(t)$，那么两者性质完全一致。但显然不可能对吧！
* 稍微可行的想法：对于任意同样的输入 $e(t)$，上述两个控制器的输出 $\hat{u}(kT) = \bar{u}(kT)$ 在采样时刻相等，那么两者性质也是接近的。到要对所有的输入都保证，依旧是很难的。
* 可行的想法：对于特殊的输入 $e(t)$，上述两个控制器的输出 $\hat{u}(kT) = \bar{u}(kT)$ 在采样时刻相等，那么两者性质也还算得上是接近的。

那么考虑哪些特殊的输入呢？（step input、impulse input）

### Step-invariance method

连续域和离散域下的阶跃响应分别如下：

$$
\hat{U}(s) = D_a(s) \frac{1}{s} \quad \text{and} \quad U(z) = D(z) \frac{z}{z - 1}
$$

希望两者在采样时刻的值是一致的，则有：

$$
D(z) \frac{z}{z - 1} = \mathcal{Z} \left[ D_a(s) \frac{1}{s} \right]
$$

因此就能够得到 Digital Controller 的表达式：

$$
D(z)  = (1-z^{-1}) \mathcal{Z} \left[ D_a(s) \frac{1}{s} \right]
$$

我们将这种离散化方法称为 **ZOH approximation method**（相当于 analog controller 和 ZOH 串联后做离散化）

* $D(z)$ 的极点除了 $(1-z^{-1})$ 和 $\mathcal{Z} \left[ D_a(s) \frac{1}{s} \right]$ 中 $\frac{1}{s}$ 带来的 $z=0$，剩余的极点都是 $D_a(s)$ 的极点通过 $z = \varepsilon^{sT}$ 映射得到的
* **$D(z)$ 与 $D_a(s)$ 的稳定性一致！**

### Impulse-invariance method

* **$D_a(s)$ is strictly proper**  

连续域下的脉冲响应，就是控制器本身的传递函数：

$$
\hat{U}(s) = D_a(s)
$$

但实际上脉冲信号是不可物理实现的，何况是在离散域下，因此脉冲信号本身是需要近似的：

$$
\delta(kT) = \begin{cases}
	\frac{1}{T}, &k = 0 \\
	0, &k = 1, 2, \dots
\end{cases} \\
\Rightarrow \mathcal{Z} \left[ \delta(t) \right] = \Sigma_{k=0}^{\infty} \delta(kT) z^{-k} = \frac{1}{T}
$$

因此，离散域下的脉冲响应为：

$$
U(z) = \frac{1}{T}D(z)
$$

希望连续域和离散域下脉冲响应在采样时刻的值是一致的，则有：

$$
\frac{1}{T}D(z) = \mathcal{Z} \left[ D_a(s) \right]
$$

因此就能够得到 Digital Controller 的表达式：

$$
D(z) = T\mathcal{Z} \left[ D_a(s) \right]
$$

* **$D_a(s)$ is not strictly proper**  

连续域下的脉冲响应，依旧就是控制器本身的传递函数：

$$
\hat{U}(s) = D_a(s) = K + \bar{D}_a(s)
$$

其中，信号 $K$ 的拉氏反变换是 $K\delta(t)$ ，该信号在时域下采样是无意义的。解决方法就是要求 $D(z)$ 的频域响应和 $D_a(s)$ 是一致的，有：

$$
D(z) = K + T\mathcal{Z} \left[ \bar{D}_a(s) \right]
$$

* $D(z)$ 的极点都是 $D_a(s)$ 的极点通过 $z = \varepsilon^{sT}$ 映射得到的
* **$D(z)$ 与 $D_a(s)$ 的稳定性一致！**



## Frequency-domain Transformation Method  

接着，可以从频域的角度来思考，如何将 Analog Controller 转换为 Digital Controller。

最简单的想法是，直接使用 s 平面与 z 平面之间的映射关系 $z = \varepsilon^{sT}$ 或 $s = \frac{ln(z)}{T}$，直接得到：

$$
D(z) \leftarrow D_a(s)|_{s = \frac{ln(z)}{T}}
$$

但是，这得到的 $D(z)$ 并不是关于 $z$ 的有理表达式，并不适合离散系统的分析。因此，可能的解决方法就是**对映射关系进行近似**。

### Forward Approximation

$$
z = \varepsilon^{sT} \approx 1 + sT \Rightarrow s \approx \frac{z-1}{T}
$$

* 为什么称为 Forward Approximation，从积分控制器的角度解释

$$
Y(s) = \frac{1}{s}E(s) \Rightarrow Y(z) = \frac{T}{z-1}E(z) = \frac{Tz^{-1}}{1-z^{-1}}E(z) \Rightarrow y(kT) = y[(k-1)T] + Te[(k-1)T]
$$

<img src="https://notes.sjtu.edu.cn/uploads/upload_292cc88902e3bcad2a4911db4e33ec07.png" style="zoom:67%;" />

* 再看看如此变换下，极点映射如何：
  
  $$
  z = 1 + sT = 1 + T(\sigma + j\omega)
  $$
  <img src="https://notes.sjtu.edu.cn/uploads/upload_55820378d2a9f2179ac9afdcca583b31.png" style="zoom: 50%;" />

  **Forward approximation 可能将一个 stable controller 𝐷(𝑠) 映射成一个 unstable 𝐷(𝑧)，不能保证稳定性一致**

### Backward Approximation

$$
z^{-1} = \varepsilon^{-sT} \approx 1 - sT \Rightarrow s \approx \frac{z-1}{Tz}
$$

* 为什么称为 Backward Approximation，从积分控制器的角度解释

$$
Y(s) = \frac{1}{s}E(s) \Rightarrow Y(z) = \frac{Tz}{z-1}E(z) = \frac{T}{1-z^{-1}}E(z) \Rightarrow y(kT) = y[(k-1)T] + Te(kT)
$$

<img src="https://notes.sjtu.edu.cn/uploads/upload_9b047c7e3ce8bee8f8a2af75b46ce68d.png" style="zoom:67%;" />

* 再看看如此变换下，极点映射如何：
  
  $$
  \begin{aligned}
  z &= \frac{1}{1 - Ts} \\
    &= \frac{1}{2} + \left[ \frac{1}{1 - Ts} - \frac{1}{2} \right] \\
    &= \frac{1}{2} - \frac{1}{2} \frac{1 + Ts}{1 - Ts} \\
  \left| z - \frac{1}{2} \right| &= \frac{1}{2} \left| \frac{1 + T\sigma + jT\omega}{1 - T\sigma - jT\omega} \right| = \frac{1}{2} \frac{\sqrt{(1 + T\sigma)^2 + T^2\omega^2}}{\sqrt{(1 - T\sigma)^2 + T^2\omega^2}}
  \end{aligned}
  $$
  <img src="https://notes.sjtu.edu.cn/uploads/upload_918d7e3eea8dee46d43d4e490da78b63.png" style="zoom:67%;" />

  **Backward approximation 可能将一个 unstable controller 𝐷(𝑠) 映射成一个 stable 𝐷(𝑧)，不能保证稳定性一致**

### Bilinear transformation  

$$
z = \varepsilon^{Ts} = \frac{\varepsilon^{sT/2}}{\varepsilon^{-sT/2}} \approx \frac{1 + (T/2)s}{1 - (T/2)s}   \Rightarrow s \approx \frac{2}{T} \frac{z - 1}{z + 1}
$$

* 双线性变换也不是第一次接触了，这里同样从积分控制器的角度来看看它是怎样的：
  
  $$
  Y(s) = \frac{1}{s}E(s) \Rightarrow Y(z) = \frac{T}{2}\frac{z+1}{z-1}E(z) = \frac{T}{2}\frac{1+z^{-1}}{1-z^{-1}}E(z) \Rightarrow y(kT) = y[(k-1)T] + \frac{T}{2}[e[(k-1)T] + e(kT)]
  $$
  <img src="https://notes.sjtu.edu.cn/uploads/upload_9876b1f687973b30ffe7d952d9db9bd5.png" style="zoom:67%;" />

* 双线性变化能够将 s 域的稳定域完美映射到 z 域的稳定域

  <img src="https://notes.sjtu.edu.cn/uploads/upload_b7bb6481df0926ff5aa280cfa13650fe.png" style="zoom:67%;" />

  **因此，Bilinear transformation 能够保证变换前后控制器稳定性一致**

### Pre-warp  

bilinear transformation  是实际中将模拟控制器变换为数字控制器用的最多的方法。但是也有一个问题，频率折叠（frequency warping）

$$
\begin{aligned}
D\left(\varepsilon^{j\omega_z T}\right) &= \left.D_a(s)\right|_{s=\frac{2}{T}\frac{\varepsilon^{j\omega_z T} - 1}{\varepsilon^{j\omega_z T} + 1}} \\
&= \left.D_a(s)\right|_{s=\frac{2}{T}\left(\frac{\varepsilon^{j\omega_z T/2} - \varepsilon^{-j\omega_z T/2}}{\varepsilon^{j\omega_z T/2} + \varepsilon^{-j\omega_z T/2}}\right)} \\
&= \left.D_a(s)\right|_{s=j\frac{2}{T}\tan\left(\frac{\omega_z T}{2}\right)}
\end{aligned}
$$

$$
\omega_s = \frac{2}{T} \tan \left( \frac{\omega_z T}{2} \right)
$$

<img src="https://notes.sjtu.edu.cn/uploads/upload_e5bc3be2ce3925d9e476ad9d5d666533.png" style="zoom:67%;" />

* 我们可以强制地将某一个特殊的频率对齐
  
  $$
  \begin{aligned}
  s &= \alpha \frac{z - 1}{z + 1} \\
  D\left(\varepsilon^{j\omega_0 T}\right) &= \left.D(s)\right|_{s=\alpha \frac{\varepsilon^{j\omega_0 T} - 1}{\varepsilon^{j\omega_0 T} + 1}} \\
  &= \left.D(s)\right|_{s=j\alpha \tan\left(\frac{\omega_0 T}{2}\right)}
  \end{aligned}
  $$

  $$
  \text{let } \omega_0 = \alpha \tan\left(\frac{\omega_0 T}{2}\right) \Rightarrow \alpha = \frac{\omega_0}{ \tan\left(\frac{\omega_0 T}{2}\right)}
  $$

  $$
  D(z) = \left.D(s)\right|_{s=\frac{\omega_0}{\tan(\omega_0 T/2)} \frac{z - 1}{z + 1}}
  $$

  **如此变换，就能够保证在特殊频率 $w_0$ 上模拟控制器和数字控制器的频率响应是一致的**



## Discretizing  State-variable Models

本文在此之前，对于系统的建模都是经典的系统框图，因此对于数字控制器的设计也都是基于经典的自动控制原理。同样，我们可以从现代控制理论的角度进行系统建模，也就是建立状态空间模型。在经典控制理论中，我们设计数字控制器，实际上是设计控制器在 z 域的表达式；在状态空间模型中，我们设计数字控制器，实际上就是**设计离散状态模型中的各个矩阵**，即

$$
\begin{aligned}
\mathbf{x}(k + 1) &= \mathbf{A}\mathbf{x}(k) + \mathbf{B}\mathbf{e}(k) \\
\hat{\mathbf{u}}(k) &= \mathbf{C}\mathbf{x}(k) + \mathbf{D}\mathbf{e}(k)
\end{aligned}
$$

中的 $A,B,C,D$ 四个参数。（这里的状态空间模型很奇怪，是从ppt上抄下来的）



同样，我们遵循一开始的思路，**将 Analog Controller 离散化成 Digital Controller**，想办法将连续的状态空间模型尽可能保持性质的离散化即可：

$$
\begin{aligned}
\dot{\mathbf{x}}(t) &= \mathbf{A}_a \mathbf{x}(t) + \mathbf{B}_a \mathbf{e}(t) \\
\hat{\mathbf{u}}(t) &= \mathbf{C}_a \mathbf{x}(t) + \mathbf{D}_a \mathbf{e}(t)
\end{aligned} \quad
\Rightarrow   \quad
\begin{aligned}
s\mathbf{X}(s) &= \mathbf{A}_a \mathbf{X}(s) + \mathbf{B}_a \mathbf{E}(s) \\
\hat{\mathbf{U}}(s) &= \mathbf{C}_a \mathbf{X}(s) + \mathbf{D}_a \mathbf{E}(s)
\end{aligned}
$$

我们可以尝试直接使用 Frequency-domain Transformation Method 对连续状态空间模型进行离散化：

* **Forward Approximation $s \approx \frac{z-1}{T}$**
  
  $$
  \begin{aligned}
  s\mathbf{X}(s) &= \mathbf{A}_a \mathbf{X}(s) + \mathbf{B}_a \mathbf{E}(s) \\
  \end{aligned} \quad
  \Rightarrow	  \quad
  \begin{aligned}
  \frac{z-1}{T}\mathbf{X}(z) &= \mathbf{A}_a \mathbf{X}(z) + \mathbf{B}_a \mathbf{E}(z) \\
  \end{aligned}
  $$

  $$
  \begin{cases}
  \mathbf{x}(k + 1) &= (\mathbf{I} + \mathbf{A}_a T) \mathbf{x}(k) + \mathbf{B}_a T \mathbf{e}(k) \\
  \hat{\mathbf{u}}(k) &= \mathbf{C}_a \mathbf{x}(k) + \mathbf{D}_a \mathbf{e}(k)
  \end{cases}
  $$

* **Backward Approximation $s \approx \frac{z-1}{Tz}$**
  
  $$
  \begin{aligned}
      \begin{aligned}
      s\mathbf{X}(s) &= \mathbf{A}_a \mathbf{X}(s) + \mathbf{B}_a \mathbf{E}(s) \\
      \end{aligned} \quad
      \Rightarrow	  \quad&
      \begin{aligned}
      \frac{z-1}{Tz}\mathbf{X}(z) &= \mathbf{A}_a \mathbf{X}(z) + \mathbf{B}_a \mathbf{E}(z) \\
      \end{aligned} \\
      \Rightarrow	  \quad&
      \begin{aligned}
      \mathbf{x}(k+1)- \mathbf{x}(k)&= \mathbf{A}_a T\mathbf{x}(k+1) + \mathbf{B}_a T\mathbf{e}(k+1) \\
  
      \end{aligned} \\
  \end{aligned}
  $$

  $$
  \text{let } \mathbf{w}(k) \triangleq (\mathbf{I} - \mathbf{A}_a T) \mathbf{x}(k) - \mathbf{B} T \mathbf{e}(k)
  $$

  $$
  \begin{cases}
  \mathbf{w}(k + 1) &= (\mathbf{I} - \mathbf{A}_a T)^{-1} \mathbf{w}(k) + (\mathbf{I} - \mathbf{A}_a T)^{-1} \mathbf{B}_a T \mathbf{e}(k) \\
  \hat{\mathbf{u}}(k) &= \mathbf{C}_a (\mathbf{I} - \mathbf{A}_a T)^{-1} \mathbf{w}(k) + [\mathbf{D}_a + \mathbf{C}_a (\mathbf{I} - \mathbf{A}_a T)^{-1} \mathbf{B}_a T] \mathbf{e}(k)
  \end{cases}
  $$

* **Bilinear Transform $s \approx \frac{2}{T} \frac{z - 1}{z + 1}$**
  
  $$
  \begin{aligned}
      \begin{aligned}
      s\mathbf{X}(s) &= \mathbf{A}_a \mathbf{X}(s) + \mathbf{B}_a \mathbf{E}(s) \\
      \end{aligned} \quad
      \Rightarrow	  \quad&
      \begin{aligned}
      \frac{2}{T} \frac{z - 1}{z + 1}\mathbf{X}(z) &= \mathbf{A}_a \mathbf{X}(z) + \mathbf{B}_a \mathbf{E}(z) \\
      \end{aligned} \\
      \Rightarrow	  \quad&
      \begin{aligned}
      2\mathbf{x}(k+1)- 2\mathbf{x}(k)&= \mathbf{A}_a T[\mathbf{x}(k+1)+\mathbf{x}(k)] + \mathbf{B}_a T[\mathbf{e}(k+1)+\mathbf{e}(k)] \\
      \end{aligned} \\
  \end{aligned}
  $$

  $$
  \text{let } \mathbf{w}(k) \triangleq \left(\mathbf{I} - \frac{\mathbf{A}_a T}{2}\right) \mathbf{x}(k) - \frac{\mathbf{B} T}{2} \mathbf{e}(k)
  $$

  $$
  \begin{cases}
  \mathbf{w}(k + 1) &= \left(\mathbf{I} + \frac{\mathbf{A}_a T}{2}\right) \left(\mathbf{I} - \frac{\mathbf{A}_a T}{2}\right)^{-1} \mathbf{w}(k) + \left(\mathbf{I} - \frac{\mathbf{A}_a T}{2}\right)^{-1} \mathbf{B}_a T \mathbf{e}(k) \\
  \hat{\mathbf{u}}(k) &= \mathbf{C}_a \left(\mathbf{I} - \frac{\mathbf{A}_a T}{2}\right)^{-1} \mathbf{w}(k) + \left[\mathbf{D}_a + \mathbf{C}_a \left(\mathbf{I} - \frac{\mathbf{A}_a T}{2}\right)^{-1} \frac{\mathbf{B}_a T}{2}\right] \mathbf{e}(k)
  \end{cases}
  $$


上述三种方法都是近似离散，实际上并不能保证前后性质的一致。

* 现代控制理论为我们提供了**将连续状态空间离散化的精确方法**：

$$
\text{if} \quad \mathbf{e}(t) = \mathbf{e}(kT), \quad kT \leq t < (k + 1)T \\
\mathbf{x}[(k + 1)T] = \boldsymbol{\Phi}(T) \mathbf{x}(kT) + \int_{kT}^{(k + 1)T} \boldsymbol{\Phi}[(k + 1)T - \tau] \mathbf{B}_a \mathbf{e}(\tau) d\tau
$$

$$
\begin{cases}
\mathbf{x}(k + 1) &= \mathbf{A}\mathbf{x}(k) + \mathbf{B}\mathbf{e}(k) \\
\hat{\mathbf{u}}(k) &= \mathbf{C}\mathbf{x}(k) + \mathbf{D}\mathbf{e}(k)
\end{cases}
$$

$$
\begin{aligned}
\mathbf{A} &= \boldsymbol{\Phi}(T), \quad \mathbf{B} = \int_{0}^{T} \boldsymbol{\Phi}(\tau) d\tau \mathbf{B}_a, \quad \mathbf{C} = \mathbf{C}_a, \quad \mathbf{D} = \mathbf{D}_a
\end{aligned}
$$



