import torch
from typing import List, Optional, Tuple
import theseus as th

class VectorDifference(th.CostFunction):
    def __init__(
        self,
        cost_weight: th.CostWeight,
        var: th.Vector,
        target: th.Vector,
        name: Optional[str] = None,
    ):
        super().__init__(cost_weight, name=name)
        self.var = var
        self.target = target
        # to improve readability, we have skipped the data checks from code block above
        self.register_optim_vars(["var"])
        self.register_aux_vars(["target"])

    def error(self) -> torch.Tensor:
        return (self.var - self.target).tensor

    def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        j = [
            # jacobian of error function wrt var is identity matrix I
            torch.eye(self.dim(), dtype=self.var.dtype)
            # repeat jacobian across each element in the batch
            .repeat(self.var.shape[0], 1, 1)
            # send to variable device
            .to(self.var.device)
        ]
        return j, self.error()

    def dim(self) -> int:
        return self.var.dof()

    def _copy_impl(self, new_name: Optional[str] = None) -> "VectorDifference":
        return VectorDifference(  # type: ignore
            self.var.copy(), self.weight.copy(), self.target.copy(), name=new_name
        )


cost_weight = th.ScaleCostWeight(1.0)

# construct cost functions and add to objective
objective = th.Objective()
num_test_fns = 10
for i in range(num_test_fns):
    a = th.Vector(2, name=f"a_{i}")
    b = th.Vector(2, name=f"b_{i}")
    cost_fn = VectorDifference(cost_weight, a, b)
    objective.add(cost_fn)

# create data for adding to the objective
theseus_inputs = {}
for i in range(num_test_fns):
    # each pair of var/target has a difference of [1, 1]
    theseus_inputs.update({f"a_{i}": torch.ones((1, 2)), f"b_{i}": 2 * torch.ones((1, 2))})

cost_fn.jacobians()
objective.update(theseus_inputs)
# sum of squares of errors [1, 1] for 10 cost fns: the result should be 20
error_sq = objective.error_squared_norm()
print(f"Sample error squared norm: {error_sq.item()}")
