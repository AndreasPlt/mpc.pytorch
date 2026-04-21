"""
Benchmark script to reproduce and measure memory leaks in MPC module.

Simulates a training loop: forward + backward through MPC on dummy data,
tracking RSS (resident set size) across epochs.
"""

import os
import gc
import torch
import torch.nn as nn
import psutil
from mpc import mpc
from mpc.mpc import QuadCost, LinDx, GradMethods


def get_rss_mb():
    """Get current process RSS in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def make_dummy_problem(n_state=4, n_ctrl=2, T=10, n_batch=8):
    """Create a dummy LQR problem for benchmarking."""
    n_tau = n_state + n_ctrl
    # Build PSD cost matrix
    _C = torch.randn(T * n_batch, n_tau, n_tau).double()
    C = (_C.transpose(1, 2).bmm(_C).view(T, n_batch, n_tau, n_tau)
         + 0.1 * torch.eye(n_tau).double().unsqueeze(0).unsqueeze(0).expand(T, n_batch, -1, -1))
    c = torch.zeros(T, n_batch, n_tau).double()

    # Linear dynamics: x_{t+1} = F @ [x_t; u_t] + f
    F = torch.randn(T - 1, n_batch, n_state, n_tau).double() * 0.1
    # Make dynamics stable
    F[:, :, :, :n_state] = 0.9 * torch.eye(n_state).double().unsqueeze(0).unsqueeze(0).expand(T - 1, n_batch, -1, -1)
    f = torch.randn(T - 1, n_batch, n_state).double() * 0.01

    x_init = torch.randn(n_batch, n_state).double()

    return C, c, F, f, x_init


def benchmark_linear_dynamics(n_epochs=50, label=""):
    """Benchmark MPC with LinDx (linear dynamics) — no neural net."""
    n_state, n_ctrl, T, n_batch = 4, 2, 10, 8

    C, c, F, f, x_init = make_dummy_problem(n_state, n_ctrl, T, n_batch)

    # Make inputs require grad so backward pass is exercised
    C = C.requires_grad_(True)
    c = c.requires_grad_(True)
    F = F.requires_grad_(True)
    f = f.requires_grad_(True)

    solver = mpc.MPC(
        n_state=n_state,
        n_ctrl=n_ctrl,
        T=T,
        lqr_iter=5,
        verbose=-1,
        exit_unconverged=False,
        detach_unconverged=False,
        backprop=True,
        n_batch=n_batch,
        u_lower=-1.0,
        u_upper=1.0,
    )

    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    rss_start = get_rss_mb()
    print(f"\n{'='*60}")
    print(f"Benchmark: LinDx {label}")
    print(f"{'='*60}")
    print(f"{'Epoch':>6} | {'RSS (MB)':>10} | {'Delta (MB)':>10} | {'Loss':>12}")
    print(f"{'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}")

    rss_prev = rss_start
    for epoch in range(n_epochs):
        # Forward
        x, u, costs = solver(x_init, QuadCost(C, c), LinDx(F, f))
        # Use u for loss (costs is detached from graph)
        loss = u.sum()

        # Backward
        loss.backward(retain_graph=False)

        # Zero grads manually (no optimizer, just measuring memory)
        if C.grad is not None:
            C.grad = None
        if c.grad is not None:
            c.grad = None
        if F.grad is not None:
            F.grad = None
        if f.grad is not None:
            f.grad = None

        gc.collect()
        rss_now = get_rss_mb()
        delta = rss_now - rss_prev

        if epoch % 5 == 0 or epoch == n_epochs - 1:
            print(f"{epoch:>6} | {rss_now:>10.1f} | {delta:>+10.1f} | {loss.item():>12.4f}")

        rss_prev = rss_now

    rss_end = get_rss_mb()
    total_growth = rss_end - rss_start
    print(f"\nTotal RSS growth: {total_growth:+.1f} MB over {n_epochs} epochs")
    print(f"Per-epoch growth: {total_growth / n_epochs:+.2f} MB/epoch")
    return total_growth


def benchmark_nn_dynamics(n_epochs=50, label=""):
    """Benchmark MPC with NNDynamics (neural net dynamics) + AUTO_DIFF."""
    from mpc.dynamics import NNDynamics

    n_state, n_ctrl, T, n_batch = 4, 2, 8, 4

    # Neural net dynamics
    dynamics = NNDynamics(n_state, n_ctrl, hidden_sizes=[32], activation='sigmoid').double()

    # Quadratic cost
    n_tau = n_state + n_ctrl
    _C = torch.randn(T * n_batch, n_tau, n_tau).double()
    C = (_C.transpose(1, 2).bmm(_C).view(T, n_batch, n_tau, n_tau)
         + 0.1 * torch.eye(n_tau).double().unsqueeze(0).unsqueeze(0).expand(T, n_batch, -1, -1))
    c = torch.zeros(T, n_batch, n_tau).double()

    x_init = torch.randn(n_batch, n_state).double()

    solver = mpc.MPC(
        n_state=n_state,
        n_ctrl=n_ctrl,
        T=T,
        lqr_iter=3,
        verbose=-1,
        exit_unconverged=False,
        detach_unconverged=False,
        backprop=True,
        n_batch=n_batch,
        grad_method=GradMethods.ANALYTIC,
    )

    optimizer = torch.optim.Adam(dynamics.parameters(), lr=1e-3)

    gc.collect()
    rss_start = get_rss_mb()
    print(f"\n{'='*60}")
    print(f"Benchmark: NNDynamics {label}")
    print(f"{'='*60}")
    print(f"{'Epoch':>6} | {'RSS (MB)':>10} | {'Delta (MB)':>10} | {'Loss':>12}")
    print(f"{'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}")

    rss_prev = rss_start
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        x, u, costs = solver(x_init, QuadCost(C, c), dynamics)
        # costs may not have grad_fn; use u (which goes through LQRStepFn)
        loss = u.sum()

        loss.backward()
        optimizer.step()

        gc.collect()
        rss_now = get_rss_mb()
        delta = rss_now - rss_prev

        if epoch % 5 == 0 or epoch == n_epochs - 1:
            print(f"{epoch:>6} | {rss_now:>10.1f} | {delta:>+10.1f} | {loss.item():>12.4f}")

        rss_prev = rss_now

    rss_end = get_rss_mb()
    total_growth = rss_end - rss_start
    print(f"\nTotal RSS growth: {total_growth:+.1f} MB over {n_epochs} epochs")
    print(f"Per-epoch growth: {total_growth / n_epochs:+.2f} MB/epoch")
    return total_growth


if __name__ == "__main__":
    print("Memory Leak Benchmark for mpc.pytorch")
    print(f"Initial RSS: {get_rss_mb():.1f} MB")

    growth_lin = benchmark_linear_dynamics(n_epochs=80, label="(constrained)")
    growth_nn = benchmark_nn_dynamics(n_epochs=80, label="(analytic grad)")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"LinDx total growth:      {growth_lin:+.1f} MB")
    print(f"NNDynamics total growth: {growth_nn:+.1f} MB")

    if growth_lin > 5 or growth_nn > 5:
        print("\n** MEMORY LEAK DETECTED **")
    else:
        print("\nMemory usage appears stable.")
