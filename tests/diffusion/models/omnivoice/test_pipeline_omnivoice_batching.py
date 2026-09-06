# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""L1 unit tests for OmniVoice request-level batching.

``OmniVoicePipeline.forward`` stacks every request's CFG halves into one
generator call, padding all of them to a single batch-wide sequence length. The
generator locates each request's target region by *offset*
(``[c_len - t_len : c_len]`` for the conditional row, ``[:t_len]`` for the
unconditional one), so the batch-wide padding is only safe as long as it lands
outside those slices, and each request's audio is only correct as long as it is
decoded from its own ``target_len`` prefix.

Both of those would fail silently — as stretched or cross-contaminated audio
rather than an exception — so they are pinned here with a pipeline shell
(``object.__new__``, the Boogu pattern) wired to deterministic fakes. No weights,
CPU only.

Also covers the batch-isolation key: seeded and voice-cloning requests must be
pinned to a batch of one.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.omnivoice.pipeline_omnivoice import (
    OmniVoicePipeline,
    _omnivoice_batch_compatibility_key,
    get_omnivoice_pre_process_func,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

_NUM_CB = 8
_MASK_ID = 1024
_REPO_ROOT = Path(__file__).resolve().parents[4]


class _RecordingGenerator:
    """Stands in for OmniVoiceGenerator; records inputs, returns known tokens.

    Mirrors the real contract: returns ``[B, 8, max_target_len]`` with each
    request's tokens left-aligned in its own row.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        target_lens = kwargs["target_lens"]
        b = len(target_lens)
        tokens = torch.zeros(b, _NUM_CB, max(target_lens), dtype=torch.long)
        # Row i is filled with its own target_len over that length, and left as
        # 0 in the batch padding beyond it. The sentinel keys on the request
        # (tests use distinct target_lens), not the row index, so batched and
        # serial runs are comparable: a decode that reads the wrong row or
        # spills into the padding shows up as a wrong value.
        for i, t_len in enumerate(target_lens):
            tokens[i, :, :t_len] = t_len
        return tokens


def _fake_decoder(tokens: torch.Tensor) -> torch.Tensor:
    """Stands in for the DAC decoder: one sample per token, value = token."""
    return tokens[:, :1, :].float()


def _make_pipeline(generator=None):
    """A pipeline shell with only what ``_generate_batch`` touches."""
    pipe = object.__new__(OmniVoicePipeline)
    pipe.device = torch.device("cpu")
    pipe.config = SimpleNamespace(num_audio_codebook=_NUM_CB, audio_mask_id=_MASK_ID)
    pipe.generator = generator if generator is not None else _RecordingGenerator()
    pipe.decoder = _fake_decoder
    pipe.num_step = 4
    pipe.guidance_scale = 2.0
    pipe.t_shift = 0.1
    pipe.layer_penalty_factor = 5.0
    pipe.position_temperature = 5.0
    pipe.class_temperature = 0.0
    return pipe


def _item(text_len: int, target_len: int, ref_len: int = 0, seed: int | None = None) -> dict:
    """Build one ``_prepare_request`` result without tokenizing anything.

    Token values are distinct per region so the assembled batch rows can be
    checked positionally: text=7, reference audio=9, target=mask.
    """
    text_ids = torch.full((_NUM_CB, text_len), 7, dtype=torch.long)
    target_ids = torch.full((_NUM_CB, target_len), _MASK_ID, dtype=torch.long)
    parts = [text_ids]
    if ref_len:
        parts.append(torch.full((_NUM_CB, ref_len), 9, dtype=torch.long))
    parts.append(target_ids)
    cond_ids = torch.cat(parts, dim=1)
    return {
        "cond_ids": cond_ids,
        "uncond_ids": target_ids.clone(),
        "cond_len": cond_ids.shape[1],
        "uncond_len": target_len,
        "text_len": text_len,
        "target_len": target_len,
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# Batch assembly
# ---------------------------------------------------------------------------


def test_generate_batch_stacks_cond_then_uncond():
    """Layout must be [cond_0..cond_{B-1}, uncond_0..uncond_{B-1}].

    The generator reads request i's unconditional row at B+i; any other
    ordering silently guides each request with another's unconditional logits.
    """
    gen = _RecordingGenerator()
    pipe = _make_pipeline(gen)
    items = [_item(text_len=3, target_len=5), _item(text_len=6, target_len=11)]

    pipe._generate_batch(items)

    ids = gen.calls[0]["input_ids"]
    max_len = max(it["cond_len"] for it in items)
    assert ids.shape == (4, _NUM_CB, max_len)
    for i, it in enumerate(items):
        # Conditional half keeps its own text prefix.
        assert torch.all(ids[i, :, : it["text_len"]] == 7)
        # Unconditional half is all-target, no text.
        assert torch.all(ids[len(items) + i, :, : it["target_len"]] == _MASK_ID)
        assert not torch.any(ids[len(items) + i] == 7)


def test_generate_batch_pads_to_batch_wide_max_len():
    """Shorter requests are right-padded; padding lands outside every slice.

    The generator slices the conditional target as ``[c_len - t_len : c_len]``,
    so padding appended *after* c_len must not shift that window.
    """
    gen = _RecordingGenerator()
    pipe = _make_pipeline(gen)
    short = _item(text_len=2, target_len=4)  # cond_len 6
    long = _item(text_len=5, target_len=20)  # cond_len 25
    items = [short, long]

    pipe._generate_batch(items)

    ids = gen.calls[0]["input_ids"]
    assert ids.shape[-1] == 25
    # The short request's real content is untouched at its own offsets...
    assert torch.all(ids[0, :, :2] == 7)
    assert torch.all(ids[0, :, 2:6] == _MASK_ID)
    # ...and the attention mask confines it to its own conditional length,
    # so the padded tail is never attended.
    attn = gen.calls[0]["attention_mask"]
    assert attn.shape == (4, 1, 25, 25)
    assert torch.all(attn[0, :, :6, :6])
    assert not torch.any(attn[0, :, 6:, :])
    assert not torch.any(attn[0, :, :, 6:])


def test_generate_batch_audio_mask_marks_reference_and_target():
    """Audio positions span [text_len, cond_len) on cond and [0, t_len) on uncond."""
    gen = _RecordingGenerator()
    pipe = _make_pipeline(gen)
    items = [_item(text_len=3, target_len=5, ref_len=2)]

    pipe._generate_batch(items)

    audio_mask = gen.calls[0]["audio_mask"]
    # text_len=3, ref=2, target=5 → cond_len 10, audio positions 3..10.
    assert not torch.any(audio_mask[0, :3])
    assert torch.all(audio_mask[0, 3:10])
    assert torch.all(audio_mask[1, :5])


def test_generate_batch_passes_every_target_len():
    """target_lens must carry all requests, not just the first."""
    gen = _RecordingGenerator()
    pipe = _make_pipeline(gen)
    items = [_item(3, 5), _item(4, 9), _item(2, 7)]

    pipe._generate_batch(items)

    assert gen.calls[0]["target_lens"] == [5, 9, 7]


# ---------------------------------------------------------------------------
# Per-request decode
# ---------------------------------------------------------------------------


def test_decode_slices_each_request_to_its_own_target_len():
    """Each request decodes its own row, trimmed to its own target length.

    Decoding the padded batch whole would run the codec over another request's
    padding and stretch this request's audio — audible, but not an error.
    """
    specs = [(3, 5), (4, 12), (2, 8)]
    audios = _make_pipeline()._generate_batch([_item(*s) for s in specs])

    assert [a.shape[-1] for a in audios] == [5, 12, 8]
    # The sentinel is each request's own target_len, so this pins that request i
    # decoded row i and never spilled into the zero padding beyond it.
    for audio, (_, target_len) in zip(audios, specs, strict=True):
        assert torch.all(audio == float(target_len))


def test_batched_output_matches_serial_calls():
    """A mixed-length B=3 batch must equal three serial B=1 calls.

    This is the property the whole optimization rests on: batching changes
    throughput, not audio.
    """
    specs = [(3, 5), (4, 12), (2, 8)]

    batched = _make_pipeline()._generate_batch([_item(*s) for s in specs])
    serial = [_make_pipeline()._generate_batch([_item(*s)])[0] for s in specs]

    assert len(batched) == len(serial)
    for got, want in zip(batched, serial, strict=True):
        torch.testing.assert_close(got, want, atol=0, rtol=0)


def test_single_request_seed_is_forwarded():
    """A batch of one keeps its seed, so seeded requests stay reproducible."""
    gen = _RecordingGenerator()
    _make_pipeline(gen)._generate_batch([_item(3, 5, seed=42)])
    assert gen.calls[0]["seed"] == 42


def test_multi_request_batch_drops_seeds():
    """The generator takes one seed per call, so a real batch must not adopt one.

    Seeded requests are pinned to batch-1 upstream; if one ever reaches a shared
    batch anyway, silently applying its seed to its neighbours is worse than
    dropping it.
    """
    gen = _RecordingGenerator()
    _make_pipeline(gen)._generate_batch([_item(3, 5, seed=42), _item(4, 9)])
    assert gen.calls[0]["seed"] is None


# ---------------------------------------------------------------------------
# Batch-isolation key
# ---------------------------------------------------------------------------


def _run_pre_process(prompt, extra_args=None):
    pre = get_omnivoice_pre_process_func(SimpleNamespace(model="k2-fsa/OmniVoice"))
    request = SimpleNamespace(
        prompt=prompt,
        request_id="req-1",
        sampling_params=SimpleNamespace(extra_args=extra_args),
        batch_compatibility_key=None,
    )
    return pre(request).batch_compatibility_key


def test_plain_tts_requests_share_a_key():
    assert _run_pre_process("hello") == _run_pre_process({"text": "hello"})
    assert _run_pre_process("hello") == _omnivoice_batch_compatibility_key(True, "ignored")


def test_capture_batch_sizes_cover_deploy_max_num_seqs(monkeypatch):
    """Every batch size the deploy profile can produce must have a pre-warmed graph.

    A gap here is silent: the shipped list was [1, 2, 4] against
    ``max_num_seqs: 4``, so a batch of 3 matched nothing and dropped to lazy
    capture — which keys on the exact seq_len rather than a bucket, capturing a
    fresh graph per distinct length on the request's critical path and thrashing
    the _MAX_LAZY_GRAPHS LRU. Request batching would have read as a regression at
    exactly the concurrency it was added for.

    The env override is cleared so this pins the shipped default, not whatever
    the developer or CI runner happens to export.
    """
    import yaml

    from vllm_omni.transformers_utils.configs.omnivoice import OmniVoiceConfig

    monkeypatch.delenv("OMNIVOICE_CUDA_GRAPH_BATCH_SIZES", raising=False)

    deploy = yaml.safe_load((_REPO_ROOT / "vllm_omni" / "deploy" / "omnivoice.yaml").read_text())
    max_num_seqs = deploy["stages"][0].get("max_num_seqs", 1)

    captured = set(OmniVoiceConfig().cuda_graph_capture_batch_sizes)
    missing = sorted(set(range(1, max_num_seqs + 1)) - captured)
    assert not missing, f"deploy max_num_seqs={max_num_seqs} can produce batch sizes {missing} with no pre-warmed graph"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1", [1]),
        ("1,2", [1, 2]),
        ("1, 2, 4", [1, 2, 4]),
        ("1 2 4", [1, 2, 4]),
    ],
    ids=["single", "comma", "comma-space", "space"],
)
def test_capture_batch_sizes_env_override(monkeypatch, raw, expected):
    """A deployment that cannot batch sets this to 1 and reclaims ~1.4 GiB of graphs."""
    from vllm_omni.transformers_utils.configs.omnivoice import OmniVoiceConfig

    monkeypatch.setenv("OMNIVOICE_CUDA_GRAPH_BATCH_SIZES", raw)
    assert OmniVoiceConfig().cuda_graph_capture_batch_sizes == expected


@pytest.mark.parametrize("raw", ["", "  ", "abc", "1,,x", "2.5"], ids=["empty", "blank", "word", "partial", "float"])
def test_capture_batch_sizes_env_override_falls_back(monkeypatch, raw):
    """A malformed value must not take the server down — it falls back to the default.

    Startup happens inside a container where a typo'd env var is hard to see; a
    working default beats a crash loop.
    """
    from vllm_omni.transformers_utils.configs.omnivoice import OmniVoiceConfig

    monkeypatch.setenv("OMNIVOICE_CUDA_GRAPH_BATCH_SIZES", raw)
    assert OmniVoiceConfig().cuda_graph_capture_batch_sizes == [1, 2, 3, 4]


def test_capture_batch_sizes_config_json_beats_env(monkeypatch):
    """Precedence is config.json > env > default, matching num_step and the TF32 knob."""
    from vllm_omni.transformers_utils.configs.omnivoice import OmniVoiceConfig

    monkeypatch.setenv("OMNIVOICE_CUDA_GRAPH_BATCH_SIZES", "1")
    cfg = OmniVoiceConfig(cuda_graph_capture_batch_sizes=[1, 2])
    assert cfg.cuda_graph_capture_batch_sizes == [1, 2]


@pytest.mark.parametrize(
    "prompt,extra",
    [
        ("hello", {"seed": 7}),
        ({"text": "hello", "ref_audio": (torch.zeros(16), 24000)}, None),
        ({"text": "hello", "multi_modal_data": {"audio": torch.zeros(16)}}, None),
    ],
    ids=["seeded", "ref_audio", "mm_audio"],
)
def test_seeded_and_cloning_requests_are_pinned_to_batch_one(prompt, extra):
    """Both must get a request-unique key.

    Seeded: the generator takes one seed for the whole batch, and
    ``test_speech_auto_voice_seed_deterministic`` asserts byte-identical audio
    for a repeated seed. Cloning: per-request reference state, not validated
    batched.
    """
    key = _run_pre_process(prompt, extra)
    assert key == _omnivoice_batch_compatibility_key(False, "req-1")
    assert key != _run_pre_process("hello")
