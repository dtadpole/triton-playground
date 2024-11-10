import torch

import triton
import triton.language as tl
import matplotlib
import pandas as pd
import numpy as np

# We can fuse `leaky_relu` by providing it as an `ACTIVATION` meta-parameter in `_matmul`.
@triton.jit
def leaky_relu(x):
    # x = x + 1
    return tl.where(x >= 0, x, 0.01 * x)

# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BLOCK_SIZE_M`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs
@triton.autotune(
    configs=[
        # triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_D': 256, 'BLOCK_SIZE_E': 64, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
        # triton.Config({'BLOCK_SIZE_B': 16, 'BLOCK_SIZE_D': 256, 'BLOCK_SIZE_E': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_D': 128, 'BLOCK_SIZE_E': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_D': 64, 'BLOCK_SIZE_E': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_B': 16, 'BLOCK_SIZE_D': 128, 'BLOCK_SIZE_E': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_D': 32, 'BLOCK_SIZE_E': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_B': 16, 'BLOCK_SIZE_D': 32, 'BLOCK_SIZE_E': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=2),
        # triton.Config({'BLOCK_SIZE_B': 16, 'BLOCK_SIZE_D': 64, 'BLOCK_SIZE_E': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=2),
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_E': 32}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_E': 32}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_E': 64}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_E': 64}, num_stages=2, num_warps=4),
        # triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_E': 64}, num_stages=2, num_warps=4),
        # triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_E': 128}, num_stages=2, num_warps=4),
        # triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_E': 128}, num_stages=2, num_warps=4),
    ],
    key=['B', 'D', 'E'],
)
@triton.jit
def mlp_wide_kernel(
    # Pointers to matrices
    x_ptr, w1_ptr, w2_ptr, o_ptr,
    # Matrix dimensions
    B, D: tl.constexpr, E,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_xb, stride_xd,
    stride_w1d, stride_w1e,
    stride_w2e, stride_w2d,
    stride_ob, stride_od,
    # Meta-parameters
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_E: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    """Kernel for computing the mlp
    Z = X @ W1, H = f(Z), O = H @ W2.
    - X has shape (B, D)
    - W1 has shape (D, E)
    - W2 has shape (E, D)
    - O has shape (B, D)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    pid_b = tl.program_id(axis=0)
    # pid_d = tl.program_id(axis=1)
    TARGET_TYPE = x_ptr.type.element_ty
    x_ptrs = tl.make_block_ptr(
        base=x_ptr,
        shape=(B, D),
        strides=(stride_xb, stride_xd),
        offsets=(pid_b * BLOCK_SIZE_B, 0),
        block_shape=(BLOCK_SIZE_B, D),
        order=(1, 0),
    )
    w1_ptrs = tl.make_block_ptr(
        base=w1_ptr,
        shape=(D, E),
        strides=(stride_w1d, stride_w1e),
        offsets=(0, 0),
        block_shape=(D, BLOCK_SIZE_E),
        order=(1, 0),
    )
    w2_ptrs = tl.make_block_ptr(
        base=w2_ptr,
        shape=(E, D),
        strides=(stride_w2e, stride_w2d),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_E, D),
        order=(1, 0),
    )
    x = tl.load(x_ptrs) # BLOCK_SIZE_B, D
    o = tl.zeros((BLOCK_SIZE_B, D), dtype=tl.float32)
    for e in range(0, tl.cdiv(E, BLOCK_SIZE_E)):
        z = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_E), dtype=tl.float32)
        # loop over D
        w1 = tl.load(w1_ptrs)       # D, BLOCK_SIZE_E
        w2 = tl.load(w2_ptrs)       # BLOCK_SIZE_E, D
        z = tl.dot(x, w1, z)        # BLOCK_SIZE_B, BLOCK_SIZE_E
        # activation
        if ACTIVATION == "leaky_relu":
            z = leaky_relu(z).to(TARGET_TYPE)
        # accumulate with o
        o = tl.dot(z, w2, o)        # BLOCK_SIZE_B, D
        # advance w1 and w2
        w1_ptrs = tl.advance(w1_ptrs, (0, BLOCK_SIZE_E))
        w2_ptrs = tl.advance(w2_ptrs, (BLOCK_SIZE_E, 0))
    o = o.to(TARGET_TYPE)
    # store o
    o_ptrs = tl.make_block_ptr(
        base=o_ptr,
        shape=(B, D),
        strides=(stride_ob, stride_od),
        offsets=(pid_b * BLOCK_SIZE_B, 0),
        block_shape=(BLOCK_SIZE_B, D),
        order=(1, 0),
    )
    tl.store(o_ptrs, o)

def mlp_wide_triton(x, w1, w2, activation=""):
    # Check constraints.
    assert x.shape[1] == w1.shape[0], "Incompatible dimensions"
    assert w1.shape[1] == w2.shape[0], "Incompatible dimensions"
    assert x.is_contiguous(), "Matrix X must be contiguous"
    assert w1.is_contiguous(), "Matrix W1 must be contiguous"
    assert w2.is_contiguous(), "Matrix W2 must be contiguous"
    B, D = x.shape
    E = w1.shape[1]

    # Allocates output.
    o = torch.empty((B, D), device=x.device, dtype=x.dtype)
    # print(o.shape)

    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(B, META['BLOCK_SIZE_B']),
    )
    mlp_wide_kernel[grid](
        x, w1, w2, o,
        B, D, E,
        x.stride(0), x.stride(1),
        w1.stride(0), w1.stride(1),
        w2.stride(0), w2.stride(1),
        o.stride(0), o.stride(1),
        ACTIVATION=activation
    )

    # print(o.shape)
    return o

def mlp_torch(x, w1, w2, activation=""):
    z = torch.matmul(x, w1)
    if activation == "leaky_relu":
        z = torch.nn.functional.leaky_relu(z)
    o = torch.matmul(z, w2)
    return o


def unit_test_simple():
    # torch.manual_seed(115)
    dtype = torch.bfloat16
    x = torch.randn((2048, 64), device='cuda', dtype=dtype)
    w1 = torch.randn((64, 1024), device='cuda', dtype=dtype)
    w2 = torch.randn((1024, 64), device='cuda', dtype=dtype)
    triton_output = mlp_wide_triton(x, w1, w2, activation="leaky_relu")
    torch_output = mlp_torch(x, w1, w2, activation="leaky_relu")
    print(f"triton_output={triton_output, triton_output[0].shape}")
    print(f"torch_output={torch_output, torch_output.shape}")
    if torch.allclose(triton_output, torch_output, atol=3e-2, rtol=1e-2):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")

    diff = np.abs(triton_output.to(torch.float32).cpu().numpy() - torch_output.to(torch.float32).cpu().numpy())
    print("max diff:",np.max(diff))
    print("mean diff:",np.mean(diff))


if __name__ == '__main__':
    DTYPE = torch.bfloat16
    HEAD = 6
    B = 1024 * HEAD
    D = 768 // HEAD
    E =768
    x = torch.randn((B, D), device='cuda', dtype=DTYPE)
    w1 = torch.randn((D, E), device='cuda', dtype=DTYPE)
    w2 = torch.randn((E, D), device='cuda', dtype=DTYPE)

    # mlp_torch(x, w1, w2, activation="leaky_relu")

    mlp_wide_triton(x, w1, w2, activation="leaky_relu")
