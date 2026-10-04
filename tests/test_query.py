import torch

from vgrep3d.query.query import _keep_dense_core, _robust_aabb


def test_keep_dense_core_rejects_distant_outlier() -> None:
    core = torch.tensor([[x, y, z] for x in (-0.1, 0.0, 0.1) for y in (-0.1, 0.1) for z in (-0.1, 0.1)])
    points = torch.cat([core, torch.tensor([[100.0, 100.0, 100.0]])])

    keep = _keep_dense_core(points)

    assert keep[:-1].all()
    assert not keep[-1]


def test_robust_aabb_clips_extreme_points() -> None:
    points = torch.cat([torch.linspace(0, 1, 101)[:, None].repeat(1, 3), torch.tensor([[50.0, 50.0, 50.0]])])

    lower, upper = _robust_aabb(points)

    assert torch.all(lower >= 0)
    assert torch.all(upper < 2)
