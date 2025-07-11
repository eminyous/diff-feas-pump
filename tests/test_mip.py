import tempfile
from collections.abc import Callable
from pathlib import Path

import gurobipy as gp
import pytest
import torch

from difffeaspump.core.mip import MIP, read

ENV = gp.Env(empty=True)
ENV.setParam("OutputFlag", 0)
ENV.start()


def make_lp() -> MIP:
    m = MIP(name="lp")
    x = m.addVar(lb=0, ub=10, vtype=gp.GRB.CONTINUOUS, name="x")
    y = m.addVar(lb=0, ub=10, vtype=gp.GRB.CONTINUOUS, name="y")
    m.addConstr(x + y <= 10, name="c1")
    m.setObjective(x + 2 * y, sense=gp.GRB.MAXIMIZE)
    m.update()
    return m


def make_ip() -> MIP:
    m = MIP(name="ip")
    x = m.addVar(lb=0, ub=10, vtype=gp.GRB.INTEGER, name="x")
    m.addConstr(x >= 3, name="c1")
    m.setObjective(x, sense=gp.GRB.MINIMIZE)
    m.update()
    return m


def make_binary() -> MIP:
    m = MIP(name="binary")
    x = m.addVar(vtype=gp.GRB.BINARY, name="x")
    y = m.addVar(vtype=gp.GRB.BINARY, name="y")
    m.addConstr(x + y == 1, name="c1")
    m.setObjective(x - y, sense=gp.GRB.MAXIMIZE)
    m.update()
    return m


def make_mip() -> MIP:
    m = MIP(name="mip")
    x = m.addVar(lb=0, ub=5, vtype=gp.GRB.INTEGER, name="x")
    y = m.addVar(lb=0, ub=1, vtype=gp.GRB.BINARY, name="y")
    z = m.addVar(lb=-1, ub=1, vtype=gp.GRB.CONTINUOUS, name="z")
    m.addConstr(x + 2 * y + z <= 7, name="c1")
    m.setObjective(x + y + z, sense=gp.GRB.MAXIMIZE)
    m.update()
    return m


def make_empty() -> MIP:
    m = MIP(name="empty")
    m.update()
    return m


def make_unbounded() -> MIP:
    m = MIP(name="unbounded")
    x = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY, name="x")
    m.setObjective(x, sense=gp.GRB.MAXIMIZE)
    m.update()
    return m


def make_infeasible() -> MIP:
    m = MIP(name="infeasible")
    x = m.addVar(lb=0, ub=1, vtype=gp.GRB.CONTINUOUS, name="x")
    m.addConstr(x >= 2, name="c1")
    m.update()
    return m


def test_lp() -> None:
    m = make_lp()
    A, b = m.Ab
    assert m.n == 2
    # 1 constraint (x + y <= 10)
    # + 2 lower bounds (x >= 0, y >= 0)
    # + 2 upper bounds (x <= 10, y <= 10) = 5
    assert m.m == 5
    assert isinstance(A, torch.Tensor)
    assert isinstance(b, torch.Tensor)
    assert m.obj.shape[0] == 2


def test_ip() -> None:
    m = make_ip()
    _A, _b = m.Ab
    assert m.n == 1
    # 1 constraint (x >= 3) + 1 lower bound (x >= 0) + 1 upper bound
    # (x <= 10) = 3
    assert m.m == 3
    assert m.integers == [0]
    assert m.binaries == []


def test_binary() -> None:
    m = make_binary()
    _A, _b = m.Ab
    assert m.n == 2
    # 1 constraint (x + y == 1)
    # + 2 lower bounds (x >= 0, y >= 0)
    # + 2 upper bounds (x <= 1, y <= 1) = 5
    assert m.m == 6
    assert set(m.binaries) == {0, 1}
    assert m.integers == []


def test_mip() -> None:
    m = make_mip()
    _A, _b = m.Ab
    assert m.n == 3
    # 1 constraint (x + 2y + z <= 7) + 3 lower bounds + 3 upper bounds = 7
    assert m.m == 7
    assert 0 in m.integers
    assert 1 in m.binaries
    assert 2 not in m.integers
    assert 2 not in m.binaries


def test_empty() -> None:
    m = make_empty()
    A, b = m.Ab
    assert m.n == 0
    assert m.m == 0
    assert A.numel() == 0
    assert b.numel() == 0


def test_unbounded() -> None:
    m = make_unbounded()
    A, b = m.Ab
    assert m.n == 1
    assert m.m == 0
    assert isinstance(A, torch.Tensor)
    assert isinstance(b, torch.Tensor)


def test_infeasible() -> None:
    m = make_infeasible()
    A, b = m.Ab
    assert m.n == 1
    # 1 constraint (x >= 2)
    # + 1 lower bound (x >= 0)
    # + 1 upper bound (x <= 1) = 3
    assert m.m == 3
    # The model is infeasible, but Ab should still be constructed
    assert isinstance(A, torch.Tensor)
    assert isinstance(b, torch.Tensor)


@pytest.mark.parametrize(
    "make_model",
    [
        make_lp,
        make_ip,
        make_binary,
        make_mip,
        make_empty,
        make_unbounded,
        make_infeasible,
    ],
)
def test_sparse(make_model: Callable[[], MIP]) -> None:
    m = make_model()
    m.sparse = True
    A, b = m.Ab
    assert hasattr(A, "is_sparse") or A.is_sparse
    assert isinstance(b, torch.Tensor)


def test_read() -> None:
    # Create a simple MIP model
    original_model = make_mip()

    # Write the model to a temporary MPS file
    with tempfile.NamedTemporaryFile(
        encoding="utf-8",
        mode="w",
        suffix=".mps",
        delete=False,
    ) as tmp_file:
        original_model.write(tmp_file.name)
        tmp_path = Path(tmp_file.name)

    try:
        # Read the model back using the read function
        loaded_model = read(str(tmp_path), env=ENV)

        # Verify that the loaded model has the same basic properties
        assert loaded_model.n == original_model.n
        assert loaded_model.m == original_model.m
        assert loaded_model.ModelName == original_model.ModelName

        # Check that variables are preserved
        original_vars = original_model.variables
        loaded_vars = loaded_model.variables
        assert len(original_vars) == len(loaded_vars)

        # Check that constraints are preserved
        original_constrs = original_model.constrs
        loaded_constrs = loaded_model.constrs
        assert len(original_constrs) == len(loaded_constrs)

        # Check that the objective is preserved
        original_obj = original_model.obj
        loaded_obj = loaded_model.obj
        assert torch.allclose(original_obj, loaded_obj)

        # Check that the constraint matrix is preserved
        original_A, original_b = original_model.Ab
        loaded_A, loaded_b = loaded_model.Ab
        assert torch.allclose(original_A, loaded_A)
        assert torch.allclose(original_b, loaded_b)

        # Check that variable types are preserved
        assert loaded_model.integers == original_model.integers
        assert loaded_model.binaries == original_model.binaries

    finally:
        # Clean up the temporary file
        if tmp_path.exists():
            tmp_path.unlink()


def test_read_nonexistent_file() -> None:
    with pytest.raises(gp.GurobiError):
        read("nonexistent_file.mps", env=ENV)
