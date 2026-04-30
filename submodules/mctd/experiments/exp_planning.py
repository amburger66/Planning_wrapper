from datasets import (
    PushBoundaryOfflineRLDataset,
    CirclePaddedOfflineRLDataset,
)
from algorithms.diffusion_forcing import DiffusionForcingPlanning
from algorithms.diffusion_forcing.df_planning_fast import (
    DiffusionForcingPlanning as DiffusionForcingPlanningFast,
)
from .exp_base import BaseLightningExperiment


class PlanningExperiment(BaseLightningExperiment):
    """
    A Partially Observed Markov Decision Process experiment
    """

    compatible_algorithms = dict(
        df_planning=DiffusionForcingPlanning,
        df_planning_fast=DiffusionForcingPlanningFast,
    )

    compatible_datasets = dict(
        # Custom PushBoundary offline trajectories.
        pushboundary_offline=PushBoundaryOfflineRLDataset,
        pushboundary_2d_offline=PushBoundaryOfflineRLDataset,
        circle_2d_offline=CirclePaddedOfflineRLDataset,
        pushblock_offline=CirclePaddedOfflineRLDataset,
    )
