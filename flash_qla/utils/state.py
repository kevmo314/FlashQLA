# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

import torch


def final_state_buffer(
    final_state_out: torch.Tensor | None,
    shape: tuple[int, ...],
    initial_state: torch.Tensor | None,
    shared_initial_state: bool,
    is_cp: bool,
    device: torch.device,
) -> torch.Tensor:
    """Where the forward kernel writes the final state: ``final_state_out`` if
    given, else a new tensor.

    ``final_state_out`` may be ``initial_state`` itself: each CTA reads its
    sequence's slice of the initial state before it writes the same slice of
    the final state. Not while every sequence reads one shared initial state
    (the first sequence's final state would overwrite it under the others),
    nor under context parallelism (a sequence's segments end on other CTAs)."""
    if final_state_out is None:
        return torch.empty(shape, dtype=torch.float32, device=device)
    if (
        final_state_out.dtype != torch.float32
        or tuple(final_state_out.shape) != tuple(shape)
        or not final_state_out.is_contiguous()
    ):
        raise ValueError(
            f"final_state_out must be a contiguous float32 tensor of shape {tuple(shape)}, "
            f"not {final_state_out.dtype} {tuple(final_state_out.shape)}"
        )
    if is_cp:
        raise ValueError("final_state_out is not supported with context parallelism")
    if (
        shared_initial_state
        and initial_state is not None
        and final_state_out.data_ptr() == initial_state.data_ptr()
    ):
        raise ValueError("final_state_out can't be a shared initial state")
    return final_state_out
