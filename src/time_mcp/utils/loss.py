import torch
from fastmcp import Context


async def get_loss_class(ctx: Context) -> torch.nn.Module:
    loss_response = await ctx.elicit(
        message="Configuring loss function:",
        response_type=["SmoothL1Loss", "MSELoss", "L1Loss", "HuberLoss"],
    )
    if loss_response.action == "accept":
        loss_type = loss_response.data
    else:
        await ctx.error(message="Loss type not provided correctly.")
        raise ValueError("Loss type not provided correctly.")
    if loss_type == "SmoothL1Loss":
        response = await ctx.elicit(
            message="Please provide the beta value for SmoothL1Loss",
            response_type=float,
        )
        if response.action == "accept":
            return torch.nn.SmoothL1Loss(beta=response.data)
        else:
            await ctx.error(message="SmoothL1Loss beta not provided correctly.")
            raise ValueError("SmoothL1Loss beta not provided correctly.")
    elif loss_type == "MSELoss":
        return torch.nn.MSELoss()
    elif loss_type == "L1Loss":
        return torch.nn.L1Loss()
    elif loss_type == "HuberLoss":
        response = await ctx.elicit(
            message="Please provide the delta value for HuberLoss", response_type=float
        )
        if response.action == "accept":
            return torch.nn.HuberLoss(delta=response.data)
        else:
            await ctx.error(message="HuberLoss delta not provided correctly.")
            raise ValueError("HuberLoss delta not provided correctly.")
    else:
        await ctx.error(message=f"Unsupported loss type: {loss_type}")
        raise ValueError(f"Unsupported loss type: {loss_type}")
