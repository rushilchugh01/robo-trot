from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None
    nn = None


_BaseModule = nn.Module if nn is not None else object


class _TXLBlock(_BaseModule):
    """Pre-norm causal attention block used by the TXL policy.

    The block receives an additive attention mask that combines causality and relative bias.
    """

    def __init__(self, d_model: int, n_head: int, d_ff: int, dropout: float) -> None:
        """Create attention and feed-forward sublayers.

        Dimensions are intentionally small by default for CPU behavior cloning.
        """
        if nn is None:
            raise ModuleNotFoundError("torch is required to construct TXL blocks")
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, current: Any, memory: Any | None, attention_mask: Any, key_padding_mask: Any | None) -> Any:
        """Run one causal query-current attention block.

        Queries are current tokens; keys/values are detached memory plus current tokens.
        """
        normalized_current = self.attn_norm(current)
        if memory is not None and int(memory.shape[1]) > 0:
            normalized_memory = self.attn_norm(memory.detach())
            key_value = torch.cat([normalized_memory, normalized_current], dim=1)
        else:
            key_value = normalized_current
        attended, _ = self.attn(
            normalized_current,
            key_value,
            key_value,
            attn_mask=attention_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        current = current + attended
        current = current + self.ff(self.ff_norm(current))
        return current


class TXLPolicy(_BaseModule):
    """Transformer-XL-style behavior-cloning policy for A1 locomotion.

    It maps `[B, T, obs_dim]` observations to tanh-bounded `[B, T, 12]` actions.
    """

    def __init__(
        self,
        obs_dim: int = 56,
        action_dim: int = 12,
        d_model: int = 128,
        n_head: int = 4,
        num_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.0,
        memory_length: int = 1000,
        max_relative_position: int = 512,
    ) -> None:
        """Create a causal segment model with detached recurrent memory.

        Memory follows Transformer-XL segment recurrence with per-layer detached caches.
        """
        if torch is None or nn is None:
            raise ModuleNotFoundError("torch is required to construct TXLPolicy")
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.d_model = int(d_model)
        self.n_head = int(n_head)
        self.num_layers = int(num_layers)
        self.d_ff = int(d_ff)
        self.dropout = float(dropout)
        self.memory_length = int(memory_length)
        self.max_relative_position = int(max_relative_position)
        self.input_proj = nn.Linear(self.obs_dim, self.d_model)
        self.blocks = nn.ModuleList(
            [_TXLBlock(self.d_model, self.n_head, self.d_ff, self.dropout) for _ in range(self.num_layers)]
        )
        self.output_norm = nn.LayerNorm(self.d_model)
        self.action_head = nn.Linear(self.d_model, self.action_dim)
        self.relative_bias = nn.Embedding(2 * self.max_relative_position + 1, self.n_head)
        self._rollout_memory: Any | None = None

    def init_memory(self, batch_size: int, device: Any | None = None) -> dict[str, Any]:
        """Return empty per-layer memories for a new sequence stream.

        The memory expands after each forward pass and tracks valid cache tokens.
        """
        if torch is None:
            raise ModuleNotFoundError("torch is required to initialize TXL memory")
        if device is None:
            device = next(self.parameters()).device
        layers = [
            torch.empty((int(batch_size), 0, self.d_model), dtype=torch.float32, device=device)
            for _ in range(self.num_layers)
        ]
        valid_mask = torch.empty((int(batch_size), 0), dtype=torch.bool, device=device)
        return {"layers": layers, "valid_mask": valid_mask}

    def forward(
        self,
        obs: Any,
        reset_mask: Any | None = None,
        memory: Any | None = None,
        valid_mask: Any | None = None,
        return_memory: bool = False,
    ) -> Any:
        """Return per-token normalized actions and optionally updated memory.

        Reset rows and padded tokens are masked so caches never cross episodes.
        """
        batch_size, sequence_length, _ = obs.shape
        memory_layers, memory_valid = self._prepare_memory(memory, reset_mask, batch_size, obs.device)
        current_valid = self._prepare_current_valid(valid_mask, batch_size, sequence_length, obs.device)
        current = self.input_proj(obs)
        new_memory: list[Any] = []
        new_valid: Any | None = None
        for layer_index, block in enumerate(self.blocks):
            layer_memory = memory_layers[layer_index] if memory_layers is not None else None
            memory_length = int(layer_memory.shape[1]) if layer_memory is not None else 0
            mask = self._attention_mask(
                batch_size=int(batch_size),
                query_length=int(sequence_length),
                key_length=int(sequence_length) + memory_length,
                memory_length=memory_length,
                device=current.device,
                dtype=current.dtype,
            )
            key_padding_mask = self._key_padding_mask(
                memory_valid=memory_valid,
                current_valid=current_valid,
                memory_length=memory_length,
                sequence_length=int(sequence_length),
                device=current.device,
                dtype=current.dtype,
            )
            current = block(current, layer_memory, mask, key_padding_mask)
            updated_layer, updated_valid = self._update_memory(layer_memory, current, memory_valid, current_valid)
            new_memory.append(updated_layer)
            new_valid = updated_valid
        action = torch.tanh(self.action_head(self.output_norm(current)))
        if return_memory:
            return action, {"layers": new_memory, "valid_mask": new_valid}
        return action

    def reset(self, rng: np.random.Generator | None = None) -> None:
        """Clear rollout memory for a new environment episode.

        The random generator argument keeps compatibility with the policy protocol.
        """
        del rng
        self._rollout_memory = None

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Run one streaming policy step and update TXL memory.

        Episode boundaries are handled by `reset()` rather than observation flags.
        """
        if torch is None:
            raise ModuleNotFoundError("torch is required to run TXLPolicy")
        self.eval()
        obs_array = np.asarray(obs, dtype=np.float32).reshape(1, 1, self.obs_dim)
        reset = np.zeros((1, 1), dtype=bool)
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs_array, dtype=torch.float32)
            reset_tensor = torch.as_tensor(reset, dtype=torch.bool)
            action, self._rollout_memory = self.forward(
                obs_tensor,
                reset_mask=reset_tensor,
                memory=self._rollout_memory,
                return_memory=True,
            )
        return action.cpu().numpy()[0, 0].astype(np.float32)

    def config_dict(self) -> dict[str, Any]:
        """Return constructor metadata for checkpoint manifests.

        Checkpoint loaders use this dictionary to rebuild architecture shapes.
        """
        return {
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "d_model": self.d_model,
            "n_head": self.n_head,
            "num_layers": self.num_layers,
            "d_ff": self.d_ff,
            "dropout": self.dropout,
            "memory_length": self.memory_length,
            "max_relative_position": self.max_relative_position,
        }

    def _prepare_memory(
        self,
        memory: Any | None,
        reset_mask: Any | None,
        batch_size: int,
        device: Any,
    ) -> tuple[list[Any] | None, Any | None]:
        """Validate memory and clear rows whose sequence begins with reset.

        Reset-aware validity prevents cached state from leaking across episodes.
        """
        if memory is None:
            return None, None
        if isinstance(memory, dict):
            layers = memory.get("layers")
            valid_mask = memory.get("valid_mask")
        else:
            layers = memory
            valid_mask = None
        if not isinstance(layers, (list, tuple)) or len(layers) != self.num_layers:
            return None, None
        if valid_mask is not None:
            valid_mask = valid_mask.to(device=device, dtype=torch.bool)
        prepared: list[Any] = []
        memory_length = int(layers[0].shape[1]) if layers else 0
        if valid_mask is None and memory_length > 0:
            valid_mask = torch.ones((int(batch_size), memory_length), dtype=torch.bool, device=device)
        if valid_mask is not None and tuple(valid_mask.shape) != (int(batch_size), memory_length):
            return None, None
        if reset_mask is not None and valid_mask is not None and memory_length > 0:
            reset_first = reset_mask[:, 0].to(device=device, dtype=torch.bool)
            if bool(torch.any(reset_first)):
                valid_mask = valid_mask.clone()
                valid_mask[reset_first] = False
        for layer_memory in layers:
            if int(layer_memory.shape[0]) != int(batch_size):
                return None, None
            if int(layer_memory.shape[1]) != memory_length:
                return None, None
            prepared.append(layer_memory)
        return prepared, valid_mask

    def _prepare_current_valid(self, valid_mask: Any | None, batch_size: int, sequence_length: int, device: Any) -> Any:
        """Return a boolean mask for valid current segment tokens.

        Missing masks mean all tokens in the current segment are valid.
        """
        if valid_mask is None:
            return torch.ones((int(batch_size), int(sequence_length)), dtype=torch.bool, device=device)
        current_valid = valid_mask.to(device=device, dtype=torch.bool)
        if tuple(current_valid.shape) != (int(batch_size), int(sequence_length)):
            return torch.ones((int(batch_size), int(sequence_length)), dtype=torch.bool, device=device)
        return current_valid

    def _update_memory(
        self,
        previous_memory: Any | None,
        current: Any,
        previous_valid: Any | None,
        current_valid: Any,
    ) -> tuple[Any, Any]:
        """Return detached memory for the next segment at one layer.

        The cache keeps hidden states plus a validity mask for attention masking.
        """
        if self.memory_length <= 0:
            return current[:, :0, :].detach(), current_valid[:, :0].detach()
        masked_current = current.masked_fill(~current_valid.unsqueeze(-1), 0.0)
        if previous_memory is not None and int(previous_memory.shape[1]) > 0:
            combined = torch.cat([previous_memory.detach(), masked_current], dim=1)
            valid = torch.cat([previous_valid, current_valid], dim=1) if previous_valid is not None else current_valid
        else:
            combined = masked_current
            valid = current_valid
        return combined[:, -self.memory_length :, :].detach(), valid[:, -self.memory_length :].detach()

    def _key_padding_mask(
        self,
        memory_valid: Any | None,
        current_valid: Any,
        memory_length: int,
        sequence_length: int,
        device: Any,
        dtype: Any,
    ) -> Any | None:
        """Return a per-row key mask for invalid memory and padded tokens.

        PyTorch masks use true values for keys that attention must ignore.
        """
        current_invalid = ~current_valid.to(device=device, dtype=torch.bool)
        if int(memory_length) > 0:
            if memory_valid is None:
                memory_invalid = torch.zeros((current_invalid.shape[0], int(memory_length)), dtype=torch.bool, device=device)
            else:
                memory_invalid = ~memory_valid.to(device=device, dtype=torch.bool)
            mask = torch.cat([memory_invalid, current_invalid], dim=1)
        else:
            mask = current_invalid
        if tuple(mask.shape) != (int(current_invalid.shape[0]), int(memory_length) + int(sequence_length)):
            return None
        if not bool(torch.any(mask)):
            return None
        additive = torch.zeros(mask.shape, dtype=dtype, device=device)
        return additive.masked_fill(mask, float("-inf"))

    def _attention_mask(
        self,
        batch_size: int,
        query_length: int,
        key_length: int,
        memory_length: int,
        device: Any,
        dtype: Any,
    ) -> Any:
        """Build a causal additive mask with learned relative-position bias.

        Position differences are clipped to `[-max_relative_position, max_relative_position]`.
        Query and key indices use token steps where cached memory precedes current tokens.
        The causal rule allows all memory keys and only current keys up to each query.
        """
        query_positions = torch.arange(int(query_length), device=device) + int(memory_length)
        key_positions = torch.arange(int(key_length), device=device)
        distance = key_positions[None, :] - query_positions[:, None]
        distance = torch.clamp(distance, -self.max_relative_position, self.max_relative_position)
        bias_index = distance + self.max_relative_position
        bias = self.relative_bias(bias_index).permute(2, 0, 1).to(dtype=dtype)
        causal_bool = key_positions[None, :] > query_positions[:, None]
        causal = torch.zeros((int(query_length), int(key_length)), device=device, dtype=dtype)
        causal = causal.masked_fill(causal_bool, float("-inf"))
        mask = bias + causal.unsqueeze(0)
        return mask.repeat(int(batch_size), 1, 1)
