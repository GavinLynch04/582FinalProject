"""
DuoAttention splits attention heads into 2 categories: retrieval and streaming. 
- Retrieval heads need full context and thus don't have any KV cache eviction.
- Streaming heads don't need full context and are only given early tokens ("attention sinks")
and the n most recent tokens. Everything else is evicted.

PyramidKV limits eviction rates based on layer depth, since early layers "funnel" attention
to important tokens in later layers.

Idea:
Apply the concept of "Pyramidal Information Funneling" from PyramidKV to DuoAttention.
Adjust the ratio of retrieval to streaming heads based on layer depth. The intuition
is that more retrieval heads higher in the network will capture important tokens and
pass them to lower layers with more streaming heads, retaining important tokens
while getting progressively lighter weight.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import torch

from kvpress.presses.duo_attention_press import DuoAttentionPress

logger = logging.getLogger(__name__)


@dataclass
class Five82Press(DuoAttentionPress):
    """
    head_compression_ratio : float
        Inherited from DuoAttentionPress, not used (in original controls % of heads made into streaming heads)
    max_retrieval_ratio : float
        % of retrieval heads in first layer
    min_retrieval_ratio : float
        % of retrieval heads in last layer
    on_the_fly_scoring : bool
        Inherited from DuoAttentionPress. If True, computes head scores on the fly instead of loading pre-computed patterns from the DuoAttention repo.
    """

    max_retrieval_ratio: float = 0.75
    min_retrieval_ratio: float = 0.05

    _pyramidal_streaming_mask: torch.Tensor = field(
        default=None, init=False, repr=False
    )

    # Override the parent's read-only compression_ratio property so the
    # experiment sweep loop can drive compression by adjusting the pyramid.
    # value=0.0 → max=1.0, min=1.0 → all heads retrieval (true uncompressed baseline)
    # value=1.0 → max=0.0, min=0.0 → all heads streaming (maximum compression)
    # min decreases faster than max so the pyramid spread grows with compression.
    # The spread at value=1.0 matches the original default spread (0.75 - 0.05 = 0.70).
    @DuoAttentionPress.compression_ratio.setter
    def compression_ratio(self, value: float):
        self.max_retrieval_ratio = max(0.0, 1.0 - value)
        self.min_retrieval_ratio = max(0.0, self.max_retrieval_ratio - value * 0.70)

    def _make_pyramid_streaming_mask(self, model):
        """
        Mask each attention head to be either a retrieval or streaming head, following pyramid pattern
        """
        # Get number of layers and number of attn heads per layer for LLM
        if self.on_the_fly_scoring:
            from kvpress.presses.duo_attention_press import duo_attention_on_the_fly
            head_scores = duo_attention_on_the_fly(model)
        else:
            _, _, head_scores = DuoAttentionPress.load_attention_pattern(model)

        num_layers, num_heads = head_scores.shape

        # Initialize mask to 0s (all retrieval heads)
        pyramid_mask = np.zeros((num_layers, num_heads), dtype=bool)

        for layer in range(num_layers):
            # Linear interpolation (get % of heads that should be retrieval based on current layer and min and max retrieval %s)
            t = layer / (num_layers - 1)
            retrieval_ratio = (
                self.max_retrieval_ratio
                + t * (self.min_retrieval_ratio - self.max_retrieval_ratio)
            )

            # Number of streaming heads at this layer
            n_streaming = round(num_heads * (1 - retrieval_ratio))

            # Lowest score heads should be streaming heads
            if n_streaming > 0:
                streaming_indices = np.argsort(head_scores[layer])[:n_streaming]
                pyramid_mask[layer, streaming_indices] = True

        return torch.tensor(pyramid_mask, dtype=torch.bool)


    def __post_init_from_model__(self, model):
        # Run DuoAttention's setup 
        super().__post_init_from_model__(model)

        # Replace streaming mask
        self.streaming_mask = self._make_pyramid_streaming_mask(model).to(model.device)