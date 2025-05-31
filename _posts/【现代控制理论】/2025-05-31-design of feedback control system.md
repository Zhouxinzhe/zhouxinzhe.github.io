---
layout:       post
title:        "【现代控制理论】- Design of Feedback Control System"
author:       "Orchid"
header-style: text
catalog:      true
hidden:       false
tags:
    - 控制
    - notes
---

上一章，我们考虑一个自治系统，介绍了李雅普诺夫稳定性。那么很自然的，如果一个自治系统本身并不稳定，有没有办法使得其变得稳定？显然只能从控制输入的角度，能否通过设计一种控制律，使得系统变得稳定。从闭环的角度思考，是不是可以设计一种闭环的控制策略？

## State Feedback Design

* **什么是状态反馈**

    <img src="https://notes.sjtu.edu.cn/uploads/upload_237f359a55ef518e7d62ffd8053d190d.png" style="zoom:67%;" />

    考虑系统
    
    $$
    \begin{cases}
    \dot{x} = Ax + Bu \\
    u = r - Kx
    \end{cases}
    $$
    
    如果 $A$ 不稳定，但是 $(A,B)$ 能控，我们可以选择合适的 $K$ 使得 $A-BK$ 稳定；如果 $A$ 的收敛速度太慢，同样可以选择合适的 $K$ 使得 $A-BK$ 稳定且加速收敛。

* **状态反馈的性质**

  * 状态反馈不改变系统的能控性
    
    $$
    \begin{align}
    G_c &= \begin{bmatrix} B & AB & \cdots & A^{n-1}B \end{bmatrix} \\
    G_{c,K} &= \begin{bmatrix} B & (A - BK)B & \cdots & (A - BK)^{n-1}B \end{bmatrix} \\
    &(A - BK)B = AB - B(KB) \\
    &(A - BK)^2B = A^2B - B(KAB) - ABKB + BKBKB \\
    &\cdots \\
    \implies& \text{rank}(G_c) = \text{rank}(G_{c,K})
    \end{align}
    $$
    
    （用人话来讲，新的能控矩阵 $G_{c,K}$ 能被原来的能控矩阵 $G_c$ 线性表示）

  * 状态反馈可能改变系统的能观性

    从传递函数的角度理解：状态反馈改变了系统的极点，但是不改变系统的零点；那么新的极点可能会和原本的零点相消，导致能观性的丢失。

  * **$A-BK$ 的特征值能够被 $K$ 任意配置，当且仅当 $(A,B)$ 能控。**
  
* 如果 $(A,B)$ 不能控

  * 首先需要进行能控性分解
  * 设计 $K$ 保证能控部分稳定；不能控的部分看命（如果是稳定的最好，不稳定的话拉倒）




## Observer Design

可能存在某些系统的状态量不能够直接得到，因此需要观测器来进行观测。

* **观测器的设计**

    <img src="https://notes.sjtu.edu.cn/uploads/upload_b3ca1e38cf317f27a1e02a97beee129b.png" style="zoom:67%;" />
    
    $$
    \begin{align}
    \text{origin system:} \quad & \dot{x} = Ax + Bu, \quad y = Cx \\
    \text{observer system:} \quad & \dot{\hat{x}} = A\hat{x} + Bu + L(y - \hat{y}), \quad \hat{y} = C\hat{x}
    \end{align}
    $$
	
	简单理解，就是用输入的误差量 $y - \hat{y}$ 来修正观测量 $\hat{x}$。我们期望状态观测的误差量 $e = x - \hat{x}$ 趋于零：
	
	$$
	\dot{e} = \dot{x} - \dot{\hat{x}} = Ax + Bu - [(A - LC)\hat{x} + LCx + Bu] \\
	= (A - LC)x - (A - LC)\hat{x} = (A - LC)e \\
	\Rightarrow \dot{e} = (A - LC)e
	$$
	
	可以将 $\dot{e} = (A - LC)e$ 看成一个自治系统，要使得该系统收敛（误差趋于零），则需要选择合适的 $L$ 使得 $A - LC$ 稳定。

* **观测器的性质**
  * **$A-LC$ 的特征值能够被 $L$ 任意配置，当且仅当 $(A,C)$ 能观。**

状态反馈的设计思路和观测器的设计思路是一致的。



## Observer-Based Feedback  

<img src="https://notes.sjtu.edu.cn/uploads/upload_616162a9c3d4f4881a301d1143b5ffc6.png" style="zoom:50%;" />
$$
\begin{align}
\text{origin system:} \quad & \dot{x} = Ax + Bu, \quad y = Cx, \quad u = r-K\hat{x} \\
\text{observer system:} \quad & \dot{\hat{x}} = A\hat{x} + Bu + L(y - \hat{y}), \quad \hat{y} = C\hat{x}
\end{align}
$$

推到可以得到：
$$
\begin{align}
\dot{x} &= Ax - BK\hat{x} + Br \\
\dot{\hat{x}} &= A\hat{x} - BK\hat{x} + Br + LC(x - \hat{x}) \\
			  &= (A-BK-LC)\hat{x} + LCx + Br
\end{align}
$$

$$
\begin{bmatrix} \dot{x} \\ \dot{\hat{x}} \end{bmatrix} = \begin{bmatrix} A & -BK \\ LC & A - BK - LC \end{bmatrix} \begin{bmatrix} x \\ \hat{x} \end{bmatrix} + \begin{bmatrix} B \\ B \end{bmatrix} r, y = \begin{bmatrix} C & 0 \end{bmatrix} \begin{bmatrix} x \\ 0 \end{bmatrix}
$$

对于这样一个复合系统，我们同样希望系统的状态是收敛的（同时希望 $\mathbf{x}, \hat{\mathbf{x}}$ 收敛），即对系统的特征值有要求：

$$
\begin{align}
& \begin{vmatrix}
z\mathbf{I} - \mathbf{A} & \mathbf{BK} \\
-\mathbf{LC} & z\mathbf{I} - \mathbf{A} + \mathbf{LC} + \mathbf{BK}
\end{vmatrix} = 0 \\
\text{列变换}\Rightarrow
& \begin{vmatrix}
z\mathbf{I} - \mathbf{A} + \mathbf{BK} & \mathbf{BK} \\
z\mathbf{I} - \mathbf{A} + \mathbf{BK} & z\mathbf{I} - \mathbf{A} + \mathbf{LC} + \mathbf{BK}
\end{vmatrix} = 0 \\
\text{行变换}\Rightarrow
&\begin{vmatrix}
z\mathbf{I} - \mathbf{A} + \mathbf{BK} & \mathbf{BK} \\
0 & z\mathbf{I} - \mathbf{A} + \mathbf{LC}
\end{vmatrix} = 0 \\
\Rightarrow
&\left| z\mathbf{I} - \mathbf{A} + \mathbf{BK} \right| \left| z\mathbf{I} - \mathbf{A} + \mathbf{LC} \right| = 0
\end{align}
$$

可以看到系统稳定的要求可以拆解成 原系统稳定 和 状态估计稳定，即 **$\mathbf{K}$ 和 $\mathbf{L}$ 的设计是解耦的**。 



## Reduced Order Observer  

考虑输出方程：

$$
y = Cx, \quad C \in R^{q \times n} 
$$

假设 $C$ 是行满秩的：

* 如果 $rank(C) = q = n$ ，$x = C^{-1}y$ ，就不需要观测器了。
* 如果 $rank(C) = q = n$，假设 $y = \begin{bmatrix} C_1 & 0 \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}, \quad C_1 \in R^{q\times q}$，那么 $x_1 = C^{-1}y$ ，仅需观测 $x_2$。

现在的问题是，如何使得 $C$ 呈现 $\begin{bmatrix} C_1 & 0 \end{bmatrix}$ 形式。

* 令 $P = \begin{bmatrix} C \\ R \end{bmatrix}^{-1} = \begin{bmatrix} P_1 & P_2 \end{bmatrix}$，其中 $R$ 是任意选择保证 $\begin{bmatrix} C \\ R \end{bmatrix} \in R^{n \times n}$ 满秩的。
  
  $$
  \begin{bmatrix} C \\ R \end{bmatrix} \begin{bmatrix} P_1 & P_2 \end{bmatrix} = \begin{bmatrix} I_q & 0 \\ 0 & I_{n-q} \end{bmatrix} \implies C \begin{bmatrix} P_1 & P_2 \end{bmatrix} = \begin{bmatrix} I_q & 0 \end{bmatrix}
  $$
	
	同时，可以得到新的状态方程：
	
	$$
	\begin{align*}
	\begin{bmatrix}
	\dot{\bar{x}}_1 \\
	\dot{\bar{x}}_2
	\end{bmatrix}
	&=
	\begin{bmatrix}
	\bar{A}_{11} & \bar{A}_{12} \\
	\bar{A}_{21} & \bar{A}_{22}
	\end{bmatrix}
	\begin{bmatrix}
	\bar{x}_1 \\
	\bar{x}_2
	\end{bmatrix}
	+
	\begin{bmatrix}
	\bar{B}_1 \\
	\bar{B}_2
	\end{bmatrix}
	u \\
	y &= \begin{bmatrix} I_q & 0 \end{bmatrix} \bar{x} = \bar{x}_1
	\end{align*}
	$$
	
	$$
	\bar{A} = P^{-1}AP, \bar{B} = P^{-1}B, \bar{C} = CP = \begin{bmatrix} I_q & 0 \end{bmatrix}
	$$
	
	$$
	\begin{align}
	    &\begin{cases}
	    \dot{\bar{x}}_1 = \bar{A}_{11} \bar{x}_1 + \bar{A}_{12} \bar{x}_2 + \bar{B}_1 u = \dot{y} \\
	    \dot{\bar{x}}_2 = \bar{A}_{22} \bar{x}_2 + \bar{A}_{21} \bar{x}_1 + \bar{B}_2 u
	    \end{cases} \\
	    \implies & \begin{cases}
	    \dot{\bar{x}}_2 = \bar{A}_{22} \bar{x}_2 + (\bar{A}_{21} y + \bar{B}_2 u) \\
		\dot{y} - \bar{A}_{11} y - \bar{B}_1 u = \bar{A}_{12} \bar{x}_2
	    \end{cases} \\
	    \text{let} &\begin{cases}
	    \bar{u} = \bar{A}_{21} y + \bar{B}_2 u \\
		\bar{y} = \dot{y} - \bar{A}_{11} y - \bar{B}_1 u
	    \end{cases} \\
	\end{align}
	$$
	
	对于状态量 $\bar{x}_2$，需要建立一个观测器进行观测：
	
	$$
	\begin{align}
	\dot{\hat{\bar{x}}}_2 &= \bar{A}_{22} \hat{\bar{x}}_2 + \bar{L}(\bar{y} - \bar{A}_{12} \hat{\bar{x}}_2) + \bar{u} \\
	&= (\bar{A}_{22} - \bar{L} \bar{A}_{12}) \hat{\bar{x}}_2 + \bar{L}(\dot{y} - \bar{A}_{11} y - \bar{B}_1 u) + \bar{A}_{21} y + \bar{B}_2 u
	\end{align}
	$$
	
	可惜的是，上述式子中出现了 $\dot{y}$，这不是我们所期望的，因为这需要额外的操作来得到。对此，我们引入一个新的状态量 $z = \hat{\bar{x}}_2 - \bar{L}y$ （实际观测器中的变量）。
	
	$$
	\begin{align}
	& \dot{\hat{\bar{x}}}_2 = (\dot{z} + \bar{L}\dot{y}) = (\bar{A}_{22} - \bar{L}\bar{A}_{12})\hat{\bar{x}}_2 + \bar{L}(\dot{y} - \bar{A}_{11}y - \bar{B}_1u) + \bar{A}_{21}y + \bar{B}_2u \\
	\implies& \begin{cases}
	\dot{z} = (\bar{A}_{22} - \bar{L}\bar{A}_{12})z + (\bar{B}_2 - \bar{L}\bar{B}_1)u + [\bar{A}_{21} - \bar{L}\bar{A}_{11} + (\bar{A}_{22} - \bar{L}\bar{A}_{12})\bar{L}]y \\
	\hat{\bar{x}}_2 = z + \bar{L}y
	\end{cases}
	\end{align}
	$$
	
	$$
	\begin{cases}
	F &= \bar{A}_{22} - \bar{L}\bar{A}_{12} \\
	G &= \bar{A}_{21} - \bar{L}\bar{A}_{11} + (\bar{A}_{22} - \bar{L}\bar{A}_{12})\bar{L} \\
	H &= \bar{B}_2 - \bar{L}\bar{B}_1 \\
	\end{cases}
	\implies
	\begin{cases}
	\dot{z} &= Fz + Gy + Hu \\
	\hat{\bar{x}}_2 &= z + \bar{L}y
	\end{cases}
	$$
	
	现在符号很多很乱哈。我们再明确一下目标，现在是想通过观测器来预测 $\bar{x}_2$，即 $\hat{\bar{x}}_2 - \bar{x}_2$ 误差趋于零，按照同样的思路：
	
	$$
	\begin{align}
	\dot{\hat{\bar{x}}}_2 - \dot{\bar{x}}_2 &= (\bar{A}_{22} \hat{\bar{x}}_2 + \bar{L}(\bar{y} - \bar{A}_{12} \hat{\bar{x}}_2) + \bar{u}) - (\bar{A}_{22} \bar{x}_2 + \bar{u}) \\
	&= \bar{A}_{22} \hat{\bar{x}}_2 - \bar{A}_{22} \bar{x}_2 + \bar{L}(\bar{A}_{12} \bar{x}_2 - \bar{A}_{12} \hat{\bar{x}}_2) \\
	&= (\bar{A}_{22} - \bar{L}\bar{A}_{12})(\hat{\bar{x}}_2 - \bar{x}_2)
	\end{align}
	$$
	
	因此，我们需要设计的就是 $\bar{L}$ ，使得 $\bar{A}_{22} - \bar{L}\bar{A}_{12}$ 的特征值均小于 0。
	
	<img src="https://notes.sjtu.edu.cn/uploads/upload_f4885bf27d86aafc116792739b20e7f8.png" style="zoom:80%;" />
	
	最后，需要通过观测器的变量 $z$ 反变换回去，得到所需的 $\hat{\bar{x}}_2$ ，再将 $\hat{\bar{x}}$ 反变换回去得到 $\hat{x}$ 。

* **总结**

  1. 选择合适的变换矩阵 $P$ ，使得 $CP = \begin{bmatrix} I_q & 0 \end{bmatrix}$

  2. 计算得到 $\bar{A} = P^{-1}AP, \bar{B} = P^{-1}B$

  3. 设计 $\bar{L}$ ，使得 $\bar{A}_{22} - \bar{L}\bar{A}_{12}$ 的特征值均小于 0

  4. 将设计得到的 $\bar{L}$ ，代入
     $$
     \begin{cases}
     F &= \bar{A}_{22} - \bar{L}\bar{A}_{12} \\
     G &= \bar{A}_{21} - \bar{L}\bar{A}_{11} + (\bar{A}_{22} - \bar{L}\bar{A}_{12})\bar{L} \\
     H &= \bar{B}_2 - \bar{L}\bar{B}_1 \\
     \end{cases}
     $$
     得到降维观测器：
     $$
     \begin{cases}
     \dot{z} &= Fz + Gy + Hu \\
     \hat{\bar{x}}_2 &= z + \bar{L}y
     \end{cases}
     $$

  5. 反向推导
     $$
     \hat{\bar{x}} = \begin{bmatrix} \bar{x}_1 \\ \hat{\bar{x}}_2 \end{bmatrix} = \begin{bmatrix} y \\ z + \bar{L}y \end{bmatrix} \\
     \hat{x} = P\hat{\bar{x}}
     $$
     最终得到观测量 $\hat{x}$ 。

      