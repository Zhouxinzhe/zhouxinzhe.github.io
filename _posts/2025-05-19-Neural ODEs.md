---
layout:       post
title:        "Neural ODEs"
author:       "Orchid"
header-style: text
catalog:      true
hidden:       false
tags:
    - Generative Models

---

## Intro

最近在学习生成模型中 Flow 这一块，其底层逻辑就是通过一步一步可逆变换，将简单的分布（比如正态分布）映射到原始数据的分布上。但是这种变换是离散的，每一层网络对应其中的一步变换，对于复杂的数据分布来说，可能需要非常多层的变换才能够得以近似。将简单的分布（比如正态分布）映射到原始数据的分布，作为生成模型的底层逻辑是没有问题的，那么是不是能够换一种方式来描述这个变换的过程？

在以往的网络中，我们对于变换过程的建模通常都是离散的，比如 Normalizing Flow 中的每一步变换、NLP 问题中使用 Transformer 离散的输出 token……但实际上，很多变换其本质上是可以连续建模的。从这个点出发，我们可以想到在物理世界中，描述一个连续变化的系统通常采用微分方程的形式，那么很自然的就会想，能不能**用神经网络来拟合微分方程，从而对于变化过程进行建模！**

这篇文章就将讲解 **Neural Ordinary Differential Equations**。本文是基于论文的个人理解，可能不能完整的揭示论文中的全部精髓。



## 微分方程

考虑以下一般的微分方程：

$$
\frac{dz}{dt} = f(z, t)
$$

我们通常考虑**解析解**和**数值解**两种求解方法。但实际情况中，解析解一般不存在。因此我们通常考虑数值解的方法，先给出一般形式：

$$
\begin{align}
&\frac{dz}{dt} = f(z, t) \\
\Rightarrow \quad &dz = f(z,t)dt \\
\Rightarrow \quad &\int_{t_0}^{t_1}dz = \int_{t_0}^{t_1}f(z,t)dt\\
\Rightarrow \quad &z(t_1) = z(t_0) + \int_{t_0}^{t_1}f(z,t)dt
\end{align}
$$

然而随之而来的问题是，计算机是不能进行积分操作的，因此需要进一步的近似。考虑泰勒展开式：

$$
z(t) = z(t_0) + \frac{t-t_0}{1!}z'(t_0) + \frac{(t-t_0)^2}{2!}z''(t_0) + \cdots
$$

接着，考虑微小步长 $\varepsilon$

$$
z(t_0 + \varepsilon) \approx z(t_0) + \varepsilon z'(t_0) = z(t_0) + \varepsilon f(z(t_0), t_0)
$$

然后通过迭代计算的方式就能求解 $z(t_1)$。上述数值求解的方式即为**欧拉法**。

* 需要注意的是，这里的步长 $\varepsilon$ 可以为正，也可以是负：当步长为正数时，迭代计算是沿着时间的正向进行的；当步长为负数时，迭代计算是沿着时间的反向进行的。因此，**理论上知道初始状态 $z(t_0)$ 以及微分方程 $f(z(t), t)$，通过数值求解可以得到任意时刻的状态。**（这一点需要清楚明白）

所以，**当我们用神经网络拟合出了微分方程 $f(z,t)$，我们仍然需要用欧拉法或者其他数值求解的方法来对变换过程进行数值求解。**



## Adjoint Sensitivity Method

对于微分方程本身的建模，你可以用简单的线性网络、MLP 等等。同时，讲解完微分方程的数值求解，用神经网络拟合微分方程的前向传播过程，就已经很清晰了。**整体的网络架构**应该是，$NeuralODE$ 中包含一个网络拟合的 $ODEF$ （微分方程），前向传播就是 $NeuralODE$ 根据 $ODEF$ 迭代计算的过程。因此关键的问题不在于，如何对微分方程建模，或是如何使用神经网络建模的微分方程进行前向传播，而是**如何进行高效的反向传播**。

为什么这么说呢？在前向传播过程中，因为数值计算涉及迭代求解，会产生大量的中间变量，如果直接采用一般的反向传播会导致高额的内存消耗（中间变量太多）以及数值计算误差（梯度累乘误差）（*原文：incurs a high memory cost and introduces additional numerical error*）。因此，该论文使用 Adjoint Sensitivity Method 来求解梯度。该方法适用于所有的微分方程数值求解方法，且对内存的消耗小，数值计算误差小（*原文：This approach scales linearly with problem size, has low memory cost, and explicitly controls numerical error*）。接下来介绍一下 **Adjoint Sensitivity Method**。（先不要管为什么，先跟着步骤一步一步来）

* 考虑优化一个标量损失函数 $L$：

    $$
    L(\mathbf{z}(t_N)) = L\left(\mathbf{z}(t_0) + \int_{t_0}^{t_N} f(\mathbf{z}(t), t, \theta) dt\right) = L\left(\text{ODESolve}(\mathbf{z}(t_0), f, t_0, t_N, \theta)\right) \tag{1}
    $$

    其中，$f$ 是网络拟合的微分方程，$\theta$ 是网络的参数（待优化的参数）。因此，**目标是求得 $\frac{\partial L}{\partial \theta}$**。

* 引入一个新的状态量 **adjoint state** $\mathbf{a}(t) = \frac{d L}{d \mathbf{z}(t)}$，尝试求解 $\frac{d\mathbf{a}(t)}{dt}$

  考虑微小步长 $\varepsilon$，我们记
  
  $$
  \mathbf{z}(t+\varepsilon) = \int_{t}^{t+\varepsilon} f(\mathbf{z}(t), t, \theta) dt + \mathbf{z}(t) = T_{\varepsilon}(\mathbf{z}(t), t) \tag{2}
  $$
  
  根据求导的链式法则
  
  $$
  \frac{dL}{d \mathbf{z}(t)} = \frac{dL}{d\mathbf{z}(t+\varepsilon)} \frac{d\mathbf{z}(t+\varepsilon)}{d\mathbf{z}(t)} \quad \Rightarrow \quad \mathbf{a}(t) = \mathbf{a}(t+\varepsilon) \frac{\partial T_{\varepsilon}(\mathbf{z}(t), t)}{\partial \mathbf{z}(t)}
  $$
  
  然后，求解$\frac{d\mathbf{a}(t)}{dt}$
  
  $$
  \begin{align}
      \frac{d\mathbf{a}(t)}{dt} &= \lim_{\varepsilon \to 0^+} \frac{\mathbf{a}(t+\varepsilon) - \mathbf{a}(t)}{\varepsilon}\\
      &= \lim_{\varepsilon \to 0^+} \frac{\mathbf{a}(t+\varepsilon) - \mathbf{a}(t+\varepsilon) \frac{\partial}{\partial \mathbf{z}(t)} T_{\varepsilon}(\mathbf{z}(t))}{\varepsilon} \quad \text{(by Eq (2))} \\
      &= \lim_{\varepsilon \to 0^+} \frac{\mathbf{a}(t+\varepsilon) - \mathbf{a}(t+\varepsilon) \frac{\partial}{\partial \mathbf{z}(t)} \left( \mathbf{z}(t) + \varepsilon f(\mathbf{z}(t), t, \theta) + \mathcal{O}(\varepsilon^2) \right)}{\varepsilon} \quad \text{(Taylor series around } \mathbf{z}(t) \text{)}\\
      &= \lim_{\varepsilon \to 0^+} \frac{\mathbf{a}(t+\varepsilon) - \mathbf{a}(t+\varepsilon) \left( I + \varepsilon \frac{\partial f(\mathbf{z}(t), t, \theta)}{\partial \mathbf{z}(t)} + \mathcal{O}(\varepsilon^2) \right)}{\varepsilon}\\
      &= \lim_{\varepsilon \to 0^+} \frac{-\varepsilon \mathbf{a}(t+\varepsilon) \frac{\partial f(\mathbf{z}(t), t, \theta)}{\partial \mathbf{z}(t)} + \mathcal{O}(\varepsilon^2)}{\varepsilon} \\
      &= \lim_{\varepsilon \to 0^+} -\mathbf{a}(t+\varepsilon) \frac{\partial f(\mathbf{z}(t), t, \theta)}{\partial \mathbf{z}(t)} + \mathcal{O}(\varepsilon)\\
      &= -\mathbf{a}(t) \frac{\partial f(\mathbf{z}(t), t, \theta)}{\partial \mathbf{z}(t)} \tag{3}
  \end{align}
  $$
  
  到这一步，已经获得了 $\mathbf{a}(t)$ 的微分方程（尽管 $\frac{\partial f(\mathbf{z}(t), t, \theta)}{\partial \mathbf{z}(t)}$ 和 $z(t)$ 及其微分方程相关）。

* 记整体网络 $NeuralODE$ 的输入为 $z(t_0)$，输出为 $z(t_N)$，损失为 $L(z(t_N))$，可以得到 $\mathbf{a}(t_N) = \frac{dL}{dz(t_N)}$。进一步可以得到任意时刻的 $\mathbf{a}(t)$ （包括 $\mathbf{a}(t_0)$）
  
    $$
    \begin{align}
    &\mathbf{z}(t + \varepsilon) = \mathbf{z}(t) + \varepsilon f(\mathbf{z}(t), t, \theta) \\
    &\mathbf{a}(t + \varepsilon) = \mathbf{a}(t) + \varepsilon (-\mathbf{a}(t) \frac{\partial f(\mathbf{z}(t), t, \theta)}{\partial \mathbf{z}(t)}) \tag{4}
    \end{align}
    $$
    
    $\mathbf{a}(t)$ 的更新计算是需要 $\mathbf{z}(t)$ 的辅助的，因为需要知道 $\frac{\partial f(\mathbf{z}(t), t, \theta)}{\partial \mathbf{z}(t)}$ 。（这就是为什么，该方法叫 adjoint（伴随） 的原由）

    论文原文中没有给出上述式子，而是直接给出了理论的积分式
    
    $$
    \begin{align*}
        \mathbf{a}(t_N) &= \frac{dL}{d\mathbf{z}(t_N)} \quad \text{initial condition of adjoint diffeq.} \\
        \mathbf{a}(t_0) &= \mathbf{a}(t_N) + \int_{t_N}^{t_0} \frac{d\mathbf{a}(t)}{dt} \, dt = \mathbf{a}(t_N) - \int_{t_N}^{t_0} \mathbf{a}(t)^T \frac{\partial f(\mathbf{z}(t), t, \theta)}{\partial \mathbf{z}(t)} \, dt \quad \text{gradient w.r.t. initial value}
    \end{align*}
    $$
    
    但在代码中，实际的运算过程是根据 $(4)$ 式的迭代过程得到的。 

到这里其实已经完整走了一遍 Adjoint Sensitivity Method，但是别忘了我们的目标是**求得 $\frac{\partial L}{\partial \theta}$**。我们目前仅仅是求得了任意时刻的 $\mathbf{a}(t) = \frac{d L}{d \mathbf{z}(t)}$，因此需要进一步的理论推导。

* 引入增广状态量 **augmented state** $Z = \begin{bmatrix} z \\ \theta \\ t \end{bmatrix} (t)$，可以得到该增广状态量的微分方程
  
  $$
  \frac{dZ}{dt} =  \frac{d}{dt} \begin{bmatrix} z \\ \theta \\ t \end{bmatrix} (t) = f_{\text{aug}}([z, \theta, t]) := \begin{bmatrix} f([z, \theta, t ]) \\ 0 \\ 1 \end{bmatrix} \tag{5}
  $$

* 同样针对该增广状态量存在一个 augmented adjoint state $\mathbf{a}_{aug}(t) = \frac{dL}{dZ}$
  
  $$
  \mathbf{a}_{\text{aug}} = \begin{bmatrix} \mathbf{a}(t) & \mathbf{a}_{\theta}(t) & \mathbf{a}_t(t) \end{bmatrix}, \quad a_{\theta}(t) := \frac{\partial L}{\partial \theta(t)}, a_t(t) := \frac{\partial L}{\partial t(t)} \tag{7}
  $$
  
  （论文原文中 $\mathbf{a}_{aug}(t)$ 定义成了列向量，应该是不正确的，$\frac{dL}{dZ}$ 应该是行向量，如果 $Z$ 是列向量的话）

  同样可以得到该 augmented adjoint state $\mathbf{a}_{aug}(t)$ 的微分方程
  
  $$
  \frac{d\mathbf{a}_{aug}(t)}{dt} = -\mathbf{a}_{aug}(t)\frac{\partial f_{aug}}{\partial Z} = -\begin{bmatrix} \mathbf{a}(t) & \mathbf{a}_{\theta}(t) & \mathbf{a}_t(t) \end{bmatrix} \frac{\partial f_{aug}}{\partial [\mathbf{z}, \theta, t]}(t) = -\begin{bmatrix} \mathbf{a} \frac{\partial f}{\partial \mathbf{z}} & \mathbf{a} \frac{\partial f}{\partial \theta} & \mathbf{a} \frac{\partial f}{\partial t} \end{bmatrix}(t)  \tag{8} \\
  $$
  
  其中，
  
  $$
  \frac{\partial f_{aug}}{\partial [\mathbf{z}, \theta, t]} = \begin{bmatrix} \frac{\partial f}{\partial \mathbf{z}} & \frac{\partial f}{\partial \theta} & \frac{\partial f}{\partial t} \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix}(t)
  $$

* 记整体网络 $NeuralODE$ 的输入为 $z(t_0)$，输出为 $z(t_N)$，损失为 $L(z(t_N))$，可以得到 $\mathbf{a}(t_N) = \frac{dL}{dz(t_N)}$、$\mathbf{a}_t(t_N) = \frac{\partial L}{\partial t(t_N)} = \frac{dL}{dz(t_N)}\frac{dz(t_N)}{dt(t_N)} = \frac{dL}{dz(t_N)}f(z(t_N),t_N)$，**令 $\mathbf{a}_{\theta}(t_N) = 0$**
  
  $$
  \begin{align}
  &\mathbf{z}(t + \varepsilon) = \mathbf{z}(t) + \varepsilon f(\mathbf{z}(t), t, \theta) \\
  &\mathbf{a}(t + \varepsilon) = \mathbf{a}(t) + \varepsilon (-\mathbf{a}(t) \frac{\partial f(\mathbf{z}(t), t, \theta)}{\partial \mathbf{z}(t)})\\
  &\mathbf{a}_{aug}(t + \varepsilon) = \mathbf{a}_{aug}(t) + \varepsilon\frac{d\mathbf{a}_{aug}(t)}{dt} \\
  \Rightarrow & \begin{bmatrix} \mathbf{a} & \mathbf{a}_{\theta} & \mathbf{a}_t \end{bmatrix}(t + \varepsilon) = \begin{bmatrix} \mathbf{a} & \mathbf{a}_{\theta} & \mathbf{a}_t \end{bmatrix}(t) + \varepsilon(-\begin{bmatrix} \mathbf{a} \frac{\partial f}{\partial \mathbf{z}} & \mathbf{a} \frac{\partial f}{\partial \theta} & \mathbf{a} \frac{\partial f}{\partial t} \end{bmatrix}(t)) \tag{9}
  \end{align}
  $$
  
  因此，从 $t = t_{N}$ 出发可以得到任意时刻的 $\mathbf{a}_{\text{aug}}(t)$。并且 $\mathbf{a}_{\text{aug}}(t)$ 的更新需要 $\mathbf{z}(t)$ 和 $\mathbf{a}(t)$ 的辅助。最终得到我们的目标 $\frac{\partial L}{\partial \theta(t_0)}$。



## Actual Application

上一小节已经完整的介绍了 Adjoint Sensitivity Method 以实现高效的反向传播，但在实际应用中还有新的扩展。在上一节中，我们考虑的标量损失函数 $L = L(\mathbf{z}(t_N))$，也就是说我们只观测了网络最后时刻的输出，并计算其损失。更高效的做法是，对于一个变换过程可以有多个观测，即在 $\mathbf{z}(t_0) \rightarrow \mathbf{z}(t_N)$ 的过程中给出更多的观测 $\left[\mathbf{z}(t_1), \mathbf{z}(t_2), \cdots, \mathbf{z}(t_N)\right]$，在计算损失的时候同时考虑变换过程中的状态，即 $L = L(\left[\mathbf{z}(t_1), \mathbf{z}(t_2), \cdots, \mathbf{z}(t_N)\right])$，这既能加速网络拟合的收敛速度，又能更好的保证网络最终收敛结果的正确性（不仅最终结果对，中间过程也要对，论文原文也是这么做的）。那么如何利用这些观测信息，得到用于更新网络参数的梯度 $\frac{\partial L}{\partial \theta}$ 呢？**划分时间段考虑！**

* 设 $\frac{\partial L}{\partial \theta} = 0$

* 考虑时间段 $t \in (t_{N-1}, t_N]$

  1. 有 $\mathbf{z}(t_N)$ 观测，同时能够得到  $\mathbf{a}(t_N) = \frac{dL}{dz(t_N)}$、$\mathbf{a}_t(t_N) = \frac{dL}{dz(t_N)}f(z(t_N),t_N)$，**令 $\mathbf{a}_{\theta}(t_N) = 0$**。

  2. 根据 $(9)$ 式迭代计算，可以得到 $\mathbf{a}_\theta(t_{N-1})$
  3. $\frac{\partial L}{\partial \theta} = \frac{\partial L}{\partial \theta} + \mathbf{a}_\theta(t_{N-1})$

* 考虑时间段 $t \in (t_{N-2}, t_{N-1}]$

  1. 有 $\mathbf{z}(t_{N-1})$ 观测，同时能够得到  $\mathbf{a}(t_{N-1}) = \frac{dL}{dz(t_{N-1})}$、$\mathbf{a}_t(t_{N-1}) = \frac{dL}{dz(t_{N-1})}f(z(t_{N-1}),t_{N-1})$，**令 $\mathbf{a}_{\theta}(t_{N-1}) = 0$**。

  2. 根据 $(9)$ 式迭代计算，可以得到 $\mathbf{a}_{\theta}(t_{N-2})$
  3. $\frac{\partial L}{\partial \theta} = \frac{\partial L}{\partial \theta} + \mathbf{a}_{\theta}(t_{N-2})$

* ……

依此类推，最终网络参数的梯度 $\frac{\partial L}{\partial \theta}$ 就是根据每个时间段 adjoint sensitivity method 反推得到的梯度的累加。



## Code

以下代码是我对参考代码做了一些简化得到的，需要在 jupyter notebook 中运行。下面是简单示例的最终拟合效果图。

<img src="https://notes.sjtu.edu.cn/uploads/upload_b18603b615552c2f91cd8cc675151c06.png" style="zoom:80%;" />

```python
import torch
from torch import Tensor
from torch import nn
from torch.nn  import functional as F 
from torch.autograd import Variable

%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from IPython.display import clear_output

import math
import numpy as np

def ode_solve(z0, t0, t1, f):
    """
    Simplest Euler ODE initial value solver
        z0 - initial state
        t0 - initial time
        t1 - final time
        f - function that computes dz/dt = f(z, t)
    """
    h_max = 0.05
    n_steps = math.ceil((abs(t1 - t0)/h_max).max().item())

    h = (t1 - t0)/n_steps
    t = t0
    z = z0

    for i_step in range(n_steps):
        z = z + h * f(z, t)
        t = t + h
    return z

class ODEF(nn.Module):
    def forward_with_grad(self, z, t, a):
        """
        Compute f and a df/dz, a df/dp, a df/dt
        -------------------------
        Arguments:
            z - state
            t - time
            a - dL/dz
        -------------------------
        Returns:
            out - f(z, t)
            adfdz - a df/dz
            adfdt - a df/dt
            adfdp - a df/dp
        """
        batch_size = z.shape[0]

        out = self.forward(z, t)

        adfdz, adfdt, *adfdp = torch.autograd.grad(
            (out,), (z, t) + tuple(self.parameters()), grad_outputs=(a),
            allow_unused=True, retain_graph=True
        )
        # grad method automatically sums gradients for batch items, we have to expand them back 
        if adfdp is not None:
            adfdp = torch.cat([p_grad.flatten() for p_grad in adfdp]).unsqueeze(0)
            adfdp = adfdp.expand(batch_size, -1) / batch_size
        if adfdt is not None:
            adfdt = adfdt.expand(batch_size, 1) / batch_size
        return out, adfdz, adfdt, adfdp

    def flatten_parameters(self):
        p_shapes = []
        flat_parameters = []
        for p in self.parameters():
            p_shapes.append(p.size())
            flat_parameters.append(p.flatten())
        return torch.cat(flat_parameters)

class ODEAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z0, t, flat_parameters, func):
        assert isinstance(func, ODEF)
        bs, *z_shape = z0.size()
        time_len = t.size(0)

        with torch.no_grad():
            z = torch.zeros(time_len, bs, *z_shape).to(z0)
            z[0] = z0
            for i_t in range(time_len - 1):
                z0 = ode_solve(z0, t[i_t], t[i_t+1], func)
                z[i_t+1] = z0

        ctx.func = func
        ctx.save_for_backward(t, z.clone(), flat_parameters)
        return z

    @staticmethod
    def backward(ctx, dLdz):
        """
        dLdz shape: time_len, batch_size, *z_shape
        """
        func = ctx.func
        t, z, flat_parameters = ctx.saved_tensors
        time_len, bs, *z_shape = z.size()
        n_dim = np.prod(z_shape)
        n_params = flat_parameters.size(0)

        # Dynamics of augmented system to be calculated backwards in time
        def augmented_dynamics(aug_z_i, t_i):
            """
            tensors here are temporal slices
            t_i - is tensor with size: bs, 1
            aug_z_i - is tensor with size: bs, n_dim*2 + n_params + 1   [z, a, a_p, a_t]
            """
            z_i, a = aug_z_i[:, :n_dim], aug_z_i[:, n_dim:2*n_dim]  # ignore parameters and time

            # Unflatten z and a
            z_i = z_i.view(bs, *z_shape)
            a = a.view(bs, *z_shape)
            with torch.set_grad_enabled(True):
                t_i = t_i.detach().requires_grad_(True)
                z_i = z_i.detach().requires_grad_(True)
                func_eval, adfdz, adfdt, adfdp = func.forward_with_grad(z_i, t_i, grad_outputs=a)  # bs, *z_shape
                adfdz = adfdz.to(z_i) if adfdz is not None else torch.zeros(bs, *z_shape).to(z_i)
                adfdp = adfdp.to(z_i) if adfdp is not None else torch.zeros(bs, n_params).to(z_i)
                adfdt = adfdt.to(z_i) if adfdt is not None else torch.zeros(bs, 1).to(z_i)

            # Flatten f and adfdz
            func_eval = func_eval.view(bs, n_dim)
            adfdz = adfdz.view(bs, n_dim) 
            return torch.cat((func_eval, -adfdz, -adfdp, -adfdt), dim=1)

        dLdz = dLdz.view(time_len, bs, n_dim)  # flatten dLdz for convenience
        with torch.no_grad():
            ## Create placeholders for output gradients
            dLdp = torch.zeros(bs, n_params).to(dLdz)

            for i_t in range(time_len-1, 0, -1):
                z_i = z[i_t]
                t_i = t[i_t]
                f_i = func(z_i, t_i).view(bs, n_dim)

                # Compute direct gradients
                dLdz_i = dLdz[i_t]
                dLdt_i = torch.bmm(torch.transpose(dLdz_i.unsqueeze(-1), 1, 2), f_i.unsqueeze(-1))[:, 0]

                # Pack augmented variable
                aug_z = torch.cat((z_i.view(bs, n_dim), dLdz_i, torch.zeros(bs, n_params).to(z), dLdt_i), dim=-1)

                # Solve augmented system backwards
                aug_ans = ode_solve(aug_z, t_i, t[i_t-1], augmented_dynamics)

                # Unpack solved backwards augmented system
                dLdp[:] += aug_ans[:, 2*n_dim:2*n_dim + n_params]

                del aug_z, aug_ans

        return None, None, dLdp, None
    
class NeuralODE(nn.Module):
    def __init__(self, func):
        super(NeuralODE, self).__init__()
        assert isinstance(func, ODEF)
        self.func = func

    def forward(self, z0, t=Tensor([0., 1.]), return_whole_sequence=False):
        t = t.to(z0)
        z = ODEAdjoint.apply(z0, t, self.func.flatten_parameters(), self.func)
        if return_whole_sequence:
            return z
        else:
            return z[-1]
        
class LinearODEF(ODEF):
    def __init__(self, W):
        super(LinearODEF, self).__init__()
        self.lin = nn.Linear(2, 2, bias=False)
        self.lin.weight = nn.Parameter(W)

    def forward(self, x, t):
        return self.lin(x)
    
class SpiralFunctionExample(LinearODEF):
    def __init__(self):
        super(SpiralFunctionExample, self).__init__(Tensor([[-0.1, -1.], [1., -0.1]]))

class RandomLinearODEF(LinearODEF):
    def __init__(self):
        super(RandomLinearODEF, self).__init__(torch.randn(2, 2)/2.)

class NNODEF(ODEF):
    def __init__(self, in_dim, hid_dim, time_invariant=False):
        super(NNODEF, self).__init__()
        self.time_invariant = time_invariant

        if time_invariant:
            self.lin1 = nn.Linear(in_dim, hid_dim)
        else:
            self.lin1 = nn.Linear(in_dim+1, hid_dim)
        self.lin2 = nn.Linear(hid_dim, hid_dim)
        self.lin3 = nn.Linear(hid_dim, in_dim)
        self.elu = nn.ELU(inplace=True)

    def forward(self, x, t):
        if not self.time_invariant:
            x = torch.cat((x, t), dim=-1)

        h = self.elu(self.lin1(x))
        h = self.elu(self.lin2(h))
        out = self.lin3(h)
        return out

def to_np(x):
    return x.detach().cpu().numpy()

def plot_trajectories(obs=None, times=None, trajs=None, save=None, figsize=(8, 4)):
    plt.figure(figsize=figsize)
    if obs is not None:
        if times is None:
            times = [None] * len(obs)
        for o, t in zip(obs, times):
            o, t = to_np(o), to_np(t)
            for b_i in range(o.shape[1]):
                plt.scatter(o[:, b_i, 0], o[:, b_i, 1], c=t[:, b_i, 0], cmap=cm.plasma)

    if trajs is not None: 
        for z in trajs:
            z = to_np(z)
            plt.plot(z[:, 0, 0], z[:, 0, 1], lw=1.5)
        if save is not None:
            plt.savefig(save)
    plt.show()

def conduct_experiment(ode_true, ode_trained, n_steps, name, plot_freq=5):
    # Create data
    z0 = Variable(torch.Tensor([[0.6, 0.3]]))

    t_max = 6.29*5
    n_points = 200

    index_np = np.arange(0, n_points, 1, dtype=np.int32)
    index_np = np.hstack([index_np[:, None]])
    times_np = np.linspace(0, t_max, num=n_points)
    times_np = np.hstack([times_np[:, None]])

    times = torch.from_numpy(times_np[:, :, None]).to(z0)
    obs = ode_true(z0, times, return_whole_sequence=True).detach()
    obs = obs + torch.randn_like(obs) * 0.01

    # Get trajectory of random timespan 
    min_delta_time = 1.0
    max_delta_time = 5.0
    max_points_num = 32
    def create_batch():
        t0 = np.random.uniform(0, t_max - max_delta_time)
        t1 = t0 + np.random.uniform(min_delta_time, max_delta_time)

        idx = sorted(np.random.permutation(index_np[(times_np > t0) & (times_np < t1)])[:max_points_num])

        obs_ = obs[idx]
        ts_ = times[idx]
        return obs_, ts_

    # Train Neural ODE
    optimizer = torch.optim.Adam(ode_trained.parameters(), lr=0.0005)
    for i in range(n_steps):
        obs_, ts_ = create_batch()

        z_ = ode_trained(obs_[0], ts_, return_whole_sequence=True)
        loss = F.mse_loss(z_, obs_.detach())

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        if i % plot_freq == 0:
            z_p = ode_trained(z0, times, return_whole_sequence=True)

            plot_trajectories(obs=[obs], times=[times], trajs=[z_p], save=None)
            clear_output(wait=True)

ode_true = NeuralODE(SpiralFunctionExample())
ode_trained = NeuralODE(NNODEF(2, 16, time_invariant=True))
conduct_experiment(ode_true, ode_trained, 1200, "linear")
```



## 相关资料

* [论文原文](https://arxiv.org/abs/1806.07366)
* [参考代码](https://github.com/msurtsukov/neural-ode)

