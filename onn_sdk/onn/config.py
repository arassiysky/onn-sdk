from __future__ import annotations

from pydantic import BaseModel


class ONNConfig(BaseModel):
    """
    Simple configuration for an ONN-based XNOR classifier.

    n:          seed length
    hidden_dim: width of MLP hidden layers
    num_layers: number of hidden layers (MLP depth)

    use_orbits:       apply L3 dihedral orbit expansion inside the model
    use_masks:        apply L4 masks inside the model
    extra_random_masks: number of additional random masks (beyond Sierpiński & Thue–Morse)
    """
    n: int
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.0

    use_orbits: bool = False
    use_masks: bool = False
    extra_random_masks: int = 0


from dataclasses import dataclass

@dataclass
class ONNConfig:
    n: int
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.0

    # Operator layer flags
    use_orbits: bool = False
    use_masks: bool = False
    extra_random_masks: int = 0

    # Multi-task settings
    multi_task: bool = False               # if True, use ONNMultiTaskModel
    num_orbit_classes: int = 4             # for orbit-size classes {1,2,3,6} → {0..3}
    lambda_bal: float = 1.0                # loss weight for balancedness
    lambda_orbit: float = 1.0              # loss weight for orbit size
