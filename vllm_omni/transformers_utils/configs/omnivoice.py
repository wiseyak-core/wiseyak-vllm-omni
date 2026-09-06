# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OmniVoice configuration for vLLM-Omni two-stage pipeline."""

import os

from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig


def _env_int_list(name: str, default: list[int]) -> list[int]:
    """Read a comma/space separated int list from the environment.

    A malformed value falls back to the default rather than failing startup: a
    typo in a deploy env var should not take the server down, and the default is
    always a working configuration.
    """
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        values = [int(part) for part in raw.replace(",", " ").split()]
    except ValueError:
        return default
    return values or default


class OmniVoiceConfig(PretrainedConfig):
    """Configuration for OmniVoice model in vLLM-Omni.

    This mirrors the HuggingFace OmniVoiceConfig but adds fields needed
    for the two-stage serving pipeline.
    """

    model_type = "omnivoice"

    def get_text_config(self, **kwargs):
        """Return self so vLLM uses our top-level config (which has
        num_attention_heads etc.) instead of trying to extract a sub-config."""
        return self

    def __init__(self, **kwargs):
        # HF repos (e.g. k2-fsa/OmniVoice) may nest generation hyperparameters.
        gen_cfg = kwargs.pop("generation_config", None)
        if isinstance(gen_cfg, dict):
            for k, v in gen_cfg.items():
                kwargs.setdefault(k, v)

        super().__init__(**kwargs)

        # Audio codec params (prefer values set by PretrainedConfig from config.json)
        self.audio_vocab_size = getattr(self, "audio_vocab_size", 1025)
        self.audio_mask_id = getattr(self, "audio_mask_id", 1024)
        self.num_audio_codebook = getattr(self, "num_audio_codebook", 8)
        self.audio_codebook_weights = getattr(
            self,
            "audio_codebook_weights",
            [8, 8, 6, 6, 4, 4, 2, 2],
        )

        # LLM backbone params (Qwen3-0.6B defaults from HF config)
        llm_config = getattr(self, "llm_config", None) or {}
        if isinstance(llm_config, PretrainedConfig):
            llm_config = llm_config.to_dict()
        elif not isinstance(llm_config, dict):
            llm_config = {}
        self.llm_hidden_size = llm_config.get("hidden_size", 1024)
        self.llm_num_hidden_layers = llm_config.get("num_hidden_layers", 28)
        self.llm_num_attention_heads = llm_config.get("num_attention_heads", 16)
        self.llm_num_key_value_heads = llm_config.get("num_key_value_heads", 8)
        self.llm_intermediate_size = llm_config.get("intermediate_size", 3072)
        self.llm_vocab_size = llm_config.get("vocab_size", 151676)
        self.llm_max_position_embeddings = llm_config.get("max_position_embeddings", 40960)
        self.llm_rope_theta = llm_config.get("rope_theta", 1000000.0)
        self.llm_rms_norm_eps = llm_config.get("rms_norm_eps", 1e-6)
        self.llm_head_dim = llm_config.get("head_dim", self.llm_hidden_size // self.llm_num_attention_heads)

        # Expose LLM params at top level for vLLM ModelConfig compatibility
        # (vLLM expects num_attention_heads, hidden_size, etc. on the config)
        self.num_attention_heads = self.llm_num_attention_heads
        self.num_key_value_heads = self.llm_num_key_value_heads
        self.num_hidden_layers = self.llm_num_hidden_layers
        self.hidden_size = self.llm_hidden_size
        self.head_dim = self.llm_head_dim
        if not hasattr(self, "vocab_size"):
            self.vocab_size = self.llm_vocab_size

        # Generation params (defaults from OmniVoiceGenerationConfig)
        # Unmasking steps. Cost is linear in this: the generator is compute
        # bound, so halving steps nearly halves GPU work per request.
        # Precedence: config.json > OMNIVOICE_NUM_STEP env var > default (32).
        # Measured on one RTX 4090 (fp16, 12-sentence Whisper-small gate):
        #   32 -> 9.90 req/s @ c=8, mean WER 0.0641   (shipped default)
        #   16 -> 19.07 req/s @ c=8, mean WER 0.0641  (1.93x, WER unchanged)
        #    8 -> 32.23 req/s @ c=8, mean WER 0.0760  (3.26x, real word errors appear)
        # 16 looks free on intelligibility, but WER does not measure prosody or
        # artifacts and output RMS does drift (0.128 -> 0.131 -> 0.158), so the
        # default stays 32; lower it deliberately after listening to the result.
        self.num_step = getattr(self, "num_step", int(os.environ.get("OMNIVOICE_NUM_STEP", "32")))
        self.guidance_scale = getattr(self, "guidance_scale", 2.0)
        self.t_shift = getattr(self, "t_shift", 0.1)
        self.layer_penalty_factor = getattr(self, "layer_penalty_factor", 5.0)
        self.position_temperature = getattr(self, "position_temperature", 5.0)
        self.class_temperature = getattr(self, "class_temperature", 0.0)

        # Audio output
        self.sample_rate = getattr(self, "sample_rate", 24000)
        self.frame_rate = getattr(self, "frame_rate", 25)

        # Serving
        self.speculative_config = None
        # Precedence: config.json > OMNIVOICE_CUDA_GRAPH env var > default (enabled).
        # Read once at init; changing OMNIVOICE_CUDA_GRAPH at runtime has no effect.
        self.enable_cuda_graph = getattr(self, "enable_cuda_graph", os.environ.get("OMNIVOICE_CUDA_GRAPH", "1") != "0")
        self.cuda_graph_capture_sizes = getattr(
            self,
            "cuda_graph_capture_sizes",
            [128, 192, 256, 320, 384, 448, 512, 640, 768, 1024],
        )
        # Request batch sizes to pre-capture graphs for. Each entry B captures
        # the CFG-doubled batch 2*B across every bucket above. Without the >1
        # entries a batched request matches no pre-warmed graph and falls into
        # lazy capture (device sync + capture under a lock) on its own critical
        # path, so request batching would read as a regression.
        #
        # This must cover *every* batch size the deploy profile's max_num_seqs
        # can produce, not just the powers of two: a miss does not round down to
        # a smaller pre-warmed batch, it drops to the lazy path, which keys on
        # the exact seq_len rather than a bucket. That captures a fresh graph per
        # distinct sequence length and thrashes the _MAX_LAZY_GRAPHS LRU.
        # Hence the contiguous 1..4 for the shipped max_num_seqs: 4.
        #
        # Each entry is not free: a graph pins a [2B, 8, L, 1025] logits buffer
        # and a [2B, 1, L, L] mask, so across the ten buckets [1, 2, 3, 4] holds
        # ~1.5 GiB against ~158 MiB for [1] alone. A deployment that cannot
        # produce batches at all — an older base image whose request lacks
        # batch_compatibility_key, so the pipeline runs unbatched — should set
        # OMNIVOICE_CUDA_GRAPH_BATCH_SIZES=1 and reclaim that.
        # Precedence: config.json > OMNIVOICE_CUDA_GRAPH_BATCH_SIZES > default.
        self.cuda_graph_capture_batch_sizes = getattr(
            self,
            "cuda_graph_capture_batch_sizes",
            _env_int_list("OMNIVOICE_CUDA_GRAPH_BATCH_SIZES", [1, 2, 3, 4]),
        )
        # TF32 matmuls: not bit-identical; opt-in (matches vLLM default-off), set OMNIVOICE_TF32=1 to enable.
        self.enable_tf32 = getattr(self, "enable_tf32", os.environ.get("OMNIVOICE_TF32", "0") != "0")


AutoConfig.register("omnivoice", OmniVoiceConfig)

__all__ = ["OmniVoiceConfig"]
