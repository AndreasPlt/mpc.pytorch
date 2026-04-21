"""Isolate the memory leak: forward-only vs forward+backward."""

import os, gc, torch, psutil
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

def run_test(name, n_epochs, do_backward=True):
    n_state, n_ctrl, T, n_batch = 4, 2, 10, 8
    C, c, F, f, x_init = make_problem(n_state, n_ctrl, T, n_batch)
    if do_backward:
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

    gc.collect()
    rss_start = get_rss_mb()
    print(f"\n--- {name} ({n_epochs} epochs) ---")

    for epoch in range(n_epochs):
        x, u, costs = solver(x_init, QuadCost(C, c), LinDx(F, f))
        if do_backward:
            u.sum().backward()
            C.grad = c.grad = F.grad = f.grad = None
        del x, u, costs
        gc.collect()

        if epoch % 20 == 0:
            print(f"  epoch {epoch:>4}: RSS={get_rss_mb():.1f} MB")

    gc.collect()
    rss_end = get_rss_mb()
    growth = rss_end - rss_start
    print(f"  TOTAL: {growth:+.1f} MB ({growth/n_epochs:+.3f} MB/epoch)")
    return growth

if __name__ == "__main__":
    run_test("Forward only (no backward)", 120, do_backward=False)
    run_test("Forward + backward", 120, do_backward=True)
