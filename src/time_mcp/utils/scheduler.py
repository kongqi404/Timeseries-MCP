import torch
from dataclasses import dataclass
from fastmcp import Context


@dataclass
class ExponentialLRConfig:
    gamma: float
    last_epoch: int = -1


@dataclass
class StepLRConfig:
    step_size: int
    gamma: float
    last_epoch: int = -1


@dataclass
class CosineAnnealingLRConfig:
    T_max: int
    eta_min: float = 0.0
    last_epoch: int = -1


async def get_scheduler(ctx: Context):
    scheduler_type_response = await ctx.elicit(
        message="Configuring learning rate scheduler:",
        response_type=["ExponentialLR", "StepLR", "CosineAnnealingLR"],
    )
    if scheduler_type_response.action == "accept":
        scheduler_type = scheduler_type_response.data
    else:
        await ctx.error(message="Scheduler type not provided correctly.")
        raise ValueError("Scheduler type not provided correctly.")
    if scheduler_type == "ExponentialLR":
        response = await ctx.elicit(
            message="Please provide the attributes for ExponentialLRConfig",
            response_type=ExponentialLRConfig,
        )
        if response.action == "accept":
            return torch.optim.lr_scheduler.ExponentialLR, response.data.__dict__
        else:
            await ctx.error(message="ExponentialLRConfig not provided correctly.")
            raise ValueError("ExponentialLRConfig not provided correctly.")
    elif scheduler_type == "StepLR":
        response = await ctx.elicit(
            message="Please provide the attributes for StepLRConfig",
            response_type=StepLRConfig,
        )
        if response.action == "accept":
            return torch.optim.lr_scheduler.StepLR, response.data.__dict__
        else:
            await ctx.error(message="StepLRConfig not provided correctly.")
            raise ValueError("StepLRConfig not provided correctly.")
    elif scheduler_type == "CosineAnnealingLR":
        response = await ctx.elicit(
            message="Please provide the attributes for CosineAnnealingLRConfig",
            response_type=CosineAnnealingLRConfig,
        )
        if response.action == "accept":
            return torch.optim.lr_scheduler.CosineAnnealingLR, response.data.__dict__
        else:
            await ctx.error(message="CosineAnnealingLRConfig not provided correctly.")
            raise ValueError("CosineAnnealingLRConfig not provided correctly.")
    else:
        await ctx.error(message=f"Unsupported scheduler type: {scheduler_type}")
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
