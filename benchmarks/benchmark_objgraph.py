"""Trace what Python objects are accumulating across epochs."""

import os, gc, torch, psutil, tracemalloc
from mpc import mpc
from mpc.mpc import QuadCost, LinDx

def get_rss_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

def make_problem(n_state=4, n_ctrl=2, T=10, n_batch=8):
    n_tau = n_state + n_ctrl
    _C = torch.randn(T * n_batch, n_tau, n_tau).double()
    C = (_C.transpose(1, 2).bmm(_C).view(T, n_batch, n_tau, n_tau)
         + 0.1 * torch.eye(n_tau).double().unsqueeze(0).unsqueeze(0).expand(T, n_batch, -1, -1))
    c = torch.zeros(T, n_batch, n_tau).double()
    F = torch.randn(T - 1, n_batch, n_state, n_tau).double() * 0.1
    F[:, :, :, :n_state] = 0.9 * torch.eye(n_state).double().unsqueeze(0).unsqueeze(0).expand(T - 1, n_batch, -1, -1)
    f = torch.randn(T - 1, n_batch, n_state).double() * 0.01
    x_init = torch.randn(n_batch, n_state).double()
    return C, c, F, f, x_init

def count_tensors():
    """Count live torch.Tensor objects."""
    gc.collect()
    count = 0
    total_size = 0
    for obj in gc.get_objects():
        try:
            if isinstance(obj, torch.Tensor):
                count += 1
                total_size += obj.nelement() * obj.element_size()
        except Exception:
            pass
    return count, total_size / 1024 / 1024

if __name__ == "__main__":
    n_state, n_ctrl, T, n_batch = 4, 2, 10, 8
    C, c, F, f, x_init = make_problem(n_state, n_ctrl, T, n_batch)
    C = C.requires_grad_(True)
    c = c.requires_grad_(True)
    F = F.requires_grad_(True)
    f = f.requires_grad_(True)

    solver = mpc.MPC(
        n_state=n_state, n_ctrl=n_ctrl, T=T,
        lqr_iter=5, verbose=-1, exit_unconverged=False,
        detach_unconverged=False, n_batch=n_batch,
        u_lower=-1.0, u_upper=1.0,
    )

    tracemalloc.start()

    print(f"{'Epoch':>5} | {'Tensors':>8} | {'Tensor MB':>10} | {'RSS MB':>8} | {'tracemalloc MB':>14}")
    print("-" * 65)

    for epoch in range(60):
        x, u, costs = solver(x_init, QuadCost(C, c), LinDx(F, f))
        u.sum().backward()
        C.grad = c.grad = F.grad = f.grad = None
        del x, u, costs

        if epoch % 5 == 0:
            gc.collect()
            n_tensors, tensor_mb = count_tensors()
            rss = get_rss_mb()
            current, peak = tracemalloc.get_traced_memory()
            print(f"{epoch:>5} | {n_tensors:>8} | {tensor_mb:>10.2f} | {rss:>8.1f} | {current/1024/1024:>14.2f}")

    # Show top allocations
    print("\nTop 10 tracemalloc allocations:")
    snapshot = tracemalloc.take_snapshot()
    stats = snapshot.statistics('lineno')
    for stat in stats[:10]:
        print(f"  {stat}")
