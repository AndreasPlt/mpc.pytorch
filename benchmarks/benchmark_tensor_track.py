"""Track exactly which tensors are accumulating and who holds them."""

import os, gc, torch, sys
from mpc import mpc
from mpc.mpc import QuadCost, LinDx

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

def get_all_tensors():
    gc.collect()
    tensors = []
    for obj in gc.get_objects():
        try:
            if isinstance(obj, torch.Tensor):
                tensors.append(obj)
        except Exception:
            pass
    return tensors

def describe_referrers(t, depth=0, seen=None, max_depth=4):
    if seen is None:
        seen = set()
    if id(t) in seen or depth > max_depth:
        return
    seen.add(id(t))
    indent = "  " * (depth + 2)
    referrers = gc.get_referrers(t)
    for ref in referrers:
        if ref is sys._getframe() or id(ref) in seen:
            continue
        ref_type = type(ref).__name__
        if ref_type == 'frame':
            continue  # Skip stack frames
        if ref_type == 'list':
            # Find who holds this list
            print(f"{indent}-> list(len={len(ref)})")
            describe_referrers(ref, depth + 1, seen, max_depth)
        elif ref_type == 'dict':
            # Find the key
            for k in list(ref.keys()):
                try:
                    if ref[k] is t:
                        print(f"{indent}-> dict[{k!r}]")
                except (KeyError, RuntimeError):
                    pass
            describe_referrers(ref, depth + 1, seen, max_depth)
        elif ref_type == 'tuple':
            print(f"{indent}-> tuple(len={len(ref)})")
            describe_referrers(ref, depth + 1, seen, max_depth)
        elif ref_type == 'cell':
            print(f"{indent}-> cell (closure variable)")
            describe_referrers(ref, depth + 1, seen, max_depth)
        elif ref_type == 'function':
            print(f"{indent}-> function: {getattr(ref, '__qualname__', ref)}")
        elif 'Backward' in ref_type or 'Node' in ref_type:
            print(f"{indent}-> autograd node: {ref_type}")
        else:
            print(f"{indent}-> {ref_type}: {repr(ref)[:100]}")


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

    # Warmup
    x, u, costs = solver(x_init, QuadCost(C, c), LinDx(F, f))
    u.sum().backward()
    C.grad = c.grad = F.grad = f.grad = None
    del x, u, costs

    before = set(id(t) for t in get_all_tensors())
    print(f"Tensors before: {len(before)}")

    # One epoch
    x, u, costs = solver(x_init, QuadCost(C, c), LinDx(F, f))
    u.sum().backward()
    C.grad = c.grad = F.grad = f.grad = None
    del x, u, costs

    after_tensors = get_all_tensors()
    after = set(id(t) for t in after_tensors)
    new_ids = after - before
    print(f"Tensors after: {len(after)}")
    print(f"New (leaked) tensors: {len(new_ids)}")

    for t in after_tensors:
        if id(t) in new_ids:
            print(f"\nLeaked tensor: shape={tuple(t.shape)}, dtype={t.dtype}, "
                  f"requires_grad={t.requires_grad}, grad_fn={type(t.grad_fn).__name__ if t.grad_fn else None}")
            describe_referrers(t)
