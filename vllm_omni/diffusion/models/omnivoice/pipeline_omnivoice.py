# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
OmniVoice TTS Pipeline for vLLM-Omni diffusion engine.

Single-stage pipeline that runs the full text-to-speech flow:
  text → tokenize → 32-step iterative unmasking → 8-codebook tokens → DAC decode → 24kHz audio

Uses request-mode execution (all steps in one forward() call).
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from typing import ClassVar

import numpy as np
import torch
from tokenizers import Tokenizer as HFTokenizer
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.interface import SupportAudioOutput
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.model_executor.models.omnivoice.duration import RuleDurationEstimator
from vllm_omni.model_executor.models.omnivoice.omnivoice_decoder import OmniVoiceDecoder
from vllm_omni.model_executor.models.omnivoice.omnivoice_generator import OmniVoiceGenerator
from vllm_omni.transformers_utils.configs.omnivoice import OmniVoiceConfig
from vllm_omni.utils.speaker_cache import get_speaker_cache

try:
    from transformers import HiggsAudioV2TokenizerModel
except ImportError:
    HiggsAudioV2TokenizerModel = None

import torchaudio

logger = init_logger(__name__)

# ==============================================================================
# Constants & Enums for OmniVoice Pipeline
# ==============================================================================


class OmniVoicePromptKey:
    """Keys used in incoming prompt dictionaries and sampling parameter extras."""
    INPUT = "input"
    TEXT = "text"
    PROMPT = "prompt"
    REF_AUDIO = "ref_audio"
    REF_TEXT = "ref_text"
    VOICE_NAME = "voice_name"
    VOICE_CREATED_AT = "voice_created_at"
    LANG = "lang"
    INSTRUCT = "instruct"
    SPEED = "speed"
    NUM_STEP = "num_step"
    CFG_STRENGTH = "cfg_strength"
    GUIDANCE_SCALE = "guidance_scale"
    SEED = "seed"
    MULTI_MODAL_DATA = "multi_modal_data"
    MM_PROCESSOR_KWARGS = "mm_processor_kwargs"
    AUDIO = "audio"


class OmniVoiceControlToken:
    """Special conditioning control tokens for the OmniVoice text tokenizer."""
    DENOISE = "<|denoise|>"
    LANG_START = "<|lang_start|>"
    LANG_END = "<|lang_end|>"
    INSTRUCT_START = "<|instruct_start|>"
    INSTRUCT_END = "<|instruct_end|>"
    TEXT_START = "<|text_start|>"
    TEXT_END = "<|text_end|>"


DEFAULT_FALLBACK_REF_TEXT: str = "Nice to meet you."
DEFAULT_FALLBACK_NUM_REF_TOKENS: int = 25
DEFAULT_FALLBACK_LANG: str = "None"
DEFAULT_FALLBACK_INSTRUCT: str = "None"
DEFAULT_SPEED_FACTOR: float = 1.0
PUNCTUATION_SENTENCE_ENDERS: tuple[str, ...] = (".", "!", "?", "।", ":", ";", "…")



def get_omnivoice_post_process_func(od_config: OmniDiffusionConfig):
    """Post-processing: convert audio tensor to numpy for WAV encoding."""

    def post_process_func(audio: torch.Tensor, output_type: str = "np"):
        if output_type == "pt":
            return audio
        return audio.cpu().float().numpy()

    return post_process_func


def _omnivoice_batch_compatibility_key(shareable: bool, request_id: str) -> tuple:
    """Request-batch isolation key.

    Plain text-to-speech requests share a key and batch together. Two cases get a
    request-unique key, which pins them to a batch of one:

    * an explicit ``seed`` — the generator takes a single seed for the whole
      batch, so a seeded request must not be co-batched or its output would stop
      being reproducible (``tests/e2e/online_serving`` asserts byte-identical
      audio for a repeated seed);
    * voice cloning — the reference audio is per-request state that has not been
      validated batched, so it fails closed rather than risking a voice leaking
      across requests.
    """
    return ("omnivoice", "tts") if shareable else ("omnivoice", "exclusive", request_id)


def get_omnivoice_pre_process_func(od_config: OmniDiffusionConfig):
    """Tag each request with the batch-isolation key described above."""

    def pre_process_func(request):
        prompt = request.prompt
        shareable = True

        extra = getattr(request.sampling_params, "extra_args", None) or {}
        if extra.get(OmniVoicePromptKey.SEED) is not None or extra.get("seed") is not None:
            shareable = False
        elif isinstance(prompt, dict):
            mm_data = prompt.get(OmniVoicePromptKey.MULTI_MODAL_DATA) or {}
            if prompt.get(OmniVoicePromptKey.REF_AUDIO) is not None or mm_data.get(OmniVoicePromptKey.AUDIO) is not None:
                shareable = False

        request.batch_compatibility_key = _omnivoice_batch_compatibility_key(shareable, request.request_id)
        return request

    return pre_process_func


# Auto-register pre_process_func into vLLM-Omni diffusion registry so batch-isolation
# ("omnivoice", "exclusive", request_id) actually executes at server runtime.
try:
    from vllm_omni.diffusion.registry import _DIFFUSION_PRE_PROCESS_FUNCS

    _DIFFUSION_PRE_PROCESS_FUNCS["OmniVoicePipeline"] = "get_omnivoice_pre_process_func"
    _DIFFUSION_PRE_PROCESS_FUNCS["OmniVoice"] = "get_omnivoice_pre_process_func"
except Exception:
    pass


def _combine_text(text, ref_text: str | None = None) -> str:
    # combine with reference text if not None
    if ref_text:
        full_text = ref_text.strip() + " " + text.strip()
    else:
        full_text = text.strip()

    # filter out newline / carriage-return characters
    full_text = re.sub(r"[\r\n]+", "", full_text)

    # replace Chinese parentheses with English ones
    full_text = full_text.replace("\uff08", "(").replace("\uff09", ")")

    # collapse consecutive spaces / tabs into a single space
    full_text = re.sub(r"[ \t]+", " ", full_text)

    # remove spaces around chinese characters
    chinese_range = r"[\u4e00-\u9fff]"
    pattern = rf"(?<={chinese_range})\s+|\s+(?={chinese_range})"
    full_text = re.sub(pattern, "", full_text)

    return full_text


_NONVERBAL_PATTERN = re.compile(
    r"\[(laughter|sigh|confirmation-en|question-en|question-ah|question-oh|"
    r"question-ei|question-yi|surprise-ah|surprise-oh|surprise-wa|"
    r"surprise-yo|dissatisfaction-hnn)\]"
)


def _tokenize_with_nonverbal_tags(text: str, tokenizer) -> list[int]:
    """Tokenize text containing non-verbal tags, handling each tag independently.

    Non-verbal tags are tokenized standalone to guarantee consistent token
    IDs regardless of surrounding language context (Chinese, English, etc.).

    Args:
        text: Full text string potentially containing non-verbal tags.
        tokenizer: HuggingFace text tokenizer instance.
    Returns:
        Token IDs list of length seq_len.
    """
    parts = []
    last_end = 0
    for m in _NONVERBAL_PATTERN.finditer(text):
        if m.start() > last_end:
            segment = text[last_end : m.start()]
            ids = tokenizer.encode(segment)
            if ids:
                parts.append(ids)
        tag_ids = tokenizer.encode(m.group())
        if tag_ids:
            parts.append(tag_ids)
        last_end = m.end()
    if last_end < len(text):
        segment = text[last_end:]
        ids = tokenizer.encode(segment)
        if ids:
            parts.append(ids)

    if not parts:
        return tokenizer.encode(text).ids
    else:
        combined = []
        for p in parts:
            combined.extend(p.ids)
    return combined


class OmniVoicePipeline(nn.Module, SupportAudioOutput):
    """OmniVoice text-to-speech pipeline for the diffusion engine.

    Wraps OmniVoiceGenerator (32-step iterative unmasking) and
    OmniVoiceDecoder (HiggsAudioV2 RVQ + DAC) into a single forward() call.
    """

    support_audio_output: ClassVar[bool] = True
    # forward() stacks every request's CFG halves into one generator call, so the
    # 32-step unmasking loop is amortized across the batch.
    supports_request_batch: ClassVar[bool] = True

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()
        self.model_path = od_config.model

        # Resolve model path (HF hub ID → local cache)
        if not os.path.isdir(self.model_path):
            from huggingface_hub import snapshot_download

            self.model_path = snapshot_download(self.model_path)

        # Load OmniVoice config
        config_path = os.path.join(self.model_path, "config.json")
        with open(config_path) as f:
            hf_config = json.load(f)
        self.config = OmniVoiceConfig(**hf_config)

        # Build generator and decoder
        self.generator = OmniVoiceGenerator(self.config)
        self.decoder = OmniVoiceDecoder(self.config)

        # Tokenizer (low-level, avoids HF tokenizer extra_special_tokens issue)
        tokenizer_path = os.path.join(self.model_path, "tokenizer.json")
        self.tokenizer = HFTokenizer.from_file(tokenizer_path)

        # Audio tokenizer for voice cloning (requires transformers>=5.3)
        if HiggsAudioV2TokenizerModel is not None:
            audio_tokenizer_path = os.path.join(self.model_path, "audio_tokenizer")
            self.audio_tokenizer = HiggsAudioV2TokenizerModel.from_pretrained(
                audio_tokenizer_path, device_map=self.device
            ).eval()
            logger.info("HiggsAudioV2 tokenizer loaded for voice cloning on %s", self.device)
        else:
            self.audio_tokenizer = None
            logger.warning("Voice cloning disabled (requires transformers>=5.3.0).")

        # Duration estimator
        self.duration_estimator = RuleDurationEstimator()

        # Speaker cache for ref_audio_tokens
        self._speaker_cache = get_speaker_cache()

        # Generation parameters
        self.num_step = self.config.num_step
        self.guidance_scale = self.config.guidance_scale
        self.t_shift = self.config.t_shift
        self.layer_penalty_factor = self.config.layer_penalty_factor
        self.position_temperature = self.config.position_temperature
        self.class_temperature = self.config.class_temperature
        self.sample_rate = self.config.sample_rate

    def _encode_ref_audio(self, audio_signal: torch.Tensor, sr: int) -> torch.Tensor:
        """Encode reference audio to 8-codebook tokens for voice cloning."""
        if self.audio_tokenizer is None:
            raise RuntimeError("Audio tokenizer not available for voice cloning")
        if audio_signal.dim() == 1:
            audio_signal = audio_signal.unsqueeze(0)
        # Resample to tokenizer's expected sample rate
        target_sr = self.audio_tokenizer.config.sample_rate
        if sr != target_sr:
            audio_signal = torchaudio.functional.resample(audio_signal, sr, target_sr)
        # Ensure mono [1, samples]
        if audio_signal.dim() == 2 and audio_signal.shape[0] > 1:
            audio_signal = audio_signal.mean(dim=0, keepdim=True)
        elif audio_signal.dim() == 1:
            audio_signal = audio_signal.unsqueeze(0)

        # RMS normalization & silence removal
        try:
            from omnivoice.utils.audio import remove_silence
            wav_np = audio_signal.cpu().numpy()
            ref_rms = float(np.sqrt(np.mean(wav_np**2)))
            if 0 < ref_rms < 0.1:
                wav_np = wav_np * 0.1 / ref_rms
            # Preserve at least 400ms trailing margin so tail phonemes/syllables of ref_text are not clipped
            wav_np = remove_silence(wav_np, target_sr, mid_sil=200, lead_sil=100, trail_sil=400)
            if wav_np.shape[-1] > 0:
                audio_signal = torch.from_numpy(wav_np).float()
        except Exception:
            try:
                wav_np = audio_signal.cpu().numpy()
                ref_rms = float(np.sqrt(np.mean(wav_np**2)))
                if 0 < ref_rms < 0.1:
                    wav_np = wav_np * 0.1 / ref_rms
                    audio_signal = torch.from_numpy(wav_np).float()
            except Exception:
                pass

        # Clip to hop_length multiple
        chunk_size = getattr(self.audio_tokenizer.config, "hop_length", None)
        if chunk_size:
            clip_size = int(audio_signal.shape[-1] % chunk_size)
            if clip_size > 0:
                audio_signal = audio_signal[:, :-clip_size]

        # Ensure mono [B, 1, samples]
        if audio_signal.dim() == 2:
            audio_signal = audio_signal.unsqueeze(1)
        with torch.inference_mode():
            tokens = self.audio_tokenizer.encode(
                audio_signal.to(self.audio_tokenizer.device), return_dict=False
            )  # [B, 8, T_ref]
            tokens = tokens.squeeze(0)  # [8, T_ref]
        return tokens

    @torch.inference_mode()
    def forward(self, req: DiffusionRequestBatch) -> list[DiffusionOutput]:
        """Generate speech audio for every request in the batch.

        Each prompt is either plain text or a structured dict:
          {"text": "...", "ref_audio": (samples, sr), "ref_text": "...",
           "lang": "...", "instruct": "..."}

        Requests are prepared independently, then their conditional and
        unconditional halves are stacked into one ``[2*B, 8, S]`` generator call
        so the 32-step unmasking loop runs once for the whole batch instead of
        once per request.
        """
        prompts = list(req.prompts) if req.prompts else [""]
        params = list(req.sampling_params_list) if req.prompts else [None]

        prepared: list[dict | None] = []
        outputs: list[DiffusionOutput | None] = [None] * len(prompts)
        for i, prompt in enumerate(prompts):
            try:
                item = self._prepare_request(prompt, params[i])
            except torch.cuda.OutOfMemoryError:
                # Not a per-request problem: the device is out of memory and the
                # rest of the batch would fail the same way. Let it propagate so
                # the worker surfaces the fault instead of reporting it as B
                # independent bad prompts and carrying on degraded.
                raise
            except Exception as exc:  # keep one bad request from failing its neighbours
                logger.exception("OmniVoice request preparation failed")
                outputs[i] = DiffusionOutput(error=str(exc))
                prepared.append(None)
                continue
            if isinstance(item, DiffusionOutput):
                outputs[i] = item
                prepared.append(None)
            else:
                prepared.append(item)

        live = [(i, p) for i, p in enumerate(prepared) if p is not None]
        if live:
            audios = self._generate_batch([p for _, p in live])
            for (i, _), audio in zip(live, audios, strict=True):
                outputs[i] = DiffusionOutput(output=audio)

        return [o if o is not None else DiffusionOutput(error="OmniVoice produced no output") for o in outputs]

    def _prepare_request(self, prompt, sampling_params) -> dict | DiffusionOutput:
        """Parse one prompt and build its generator inputs.

        Returns a dict of per-request tensors, or a ``DiffusionOutput`` carrying
        a user-facing error for a prompt that cannot be synthesized.
        """
        ref_audio = None
        ref_text = None
        lang = DEFAULT_FALLBACK_LANG
        instruct = DEFAULT_FALLBACK_INSTRUCT
        extra = (getattr(sampling_params, "extra_args", None) or {}) if sampling_params else {}
        seed = extra.get(OmniVoicePromptKey.SEED, None)

        voice_name = None
        if isinstance(prompt, dict):
            # Top-level keys (used by serving_speech.py /v1/audio/speech path)
            text = prompt.get(OmniVoicePromptKey.INPUT) or prompt.get(OmniVoicePromptKey.TEXT) or prompt.get(OmniVoicePromptKey.PROMPT)
            ref_audio = prompt.get(OmniVoicePromptKey.REF_AUDIO)
            ref_text = prompt.get(OmniVoicePromptKey.REF_TEXT)
            voice_name = prompt.get(OmniVoicePromptKey.VOICE_NAME)
            lang = prompt.get(OmniVoicePromptKey.LANG)
            instruct = prompt.get(OmniVoicePromptKey.INSTRUCT)
            # OmniTextPrompt format (used by offline Omni.generate path):
            # ref_audio comes via multi_modal_data["audio"] and the rest via
            # mm_processor_kwargs. Fall back to those when top-level keys are
            # absent so both invocation styles work.
            mm_data = prompt.get(OmniVoicePromptKey.MULTI_MODAL_DATA) or {}
            mm_kwargs = prompt.get(OmniVoicePromptKey.MM_PROCESSOR_KWARGS) or {}
            if ref_audio is None:
                audio_field = mm_data.get(OmniVoicePromptKey.AUDIO)
                # Standard multimodal shape allows a list of audios; OmniVoice
                # voice cloning conditions on a single reference clip, so
                # unwrap a length-1 list and reject multi-reference prompts up
                # front (otherwise a list would later crash inside
                # ``_encode_ref_audio`` when it calls ``audio.dim()``).
                if isinstance(audio_field, list):
                    if len(audio_field) == 1:
                        audio_field = audio_field[0]
                    elif len(audio_field) > 1:
                        return DiffusionOutput(
                            error=f"OmniVoice voice cloning supports a single reference audio; got {len(audio_field)}"  # noqa: E501
                        )
                    else:
                        audio_field = None
                if audio_field is not None:
                    if isinstance(audio_field, tuple) and len(audio_field) == 2:
                        ref_audio = audio_field
                    else:
                        sr = mm_kwargs.get("sample_rate") or self.sample_rate
                        ref_audio = (audio_field, int(sr))
            if ref_text is None:
                ref_text = mm_kwargs.get(OmniVoicePromptKey.REF_TEXT)
            if lang is None:
                lang = mm_kwargs.get(OmniVoicePromptKey.LANG)
            if instruct is None:
                instruct = mm_kwargs.get(OmniVoicePromptKey.INSTRUCT)

            if not text:
                return DiffusionOutput(error="Empty text prompt")
            lang = lang or DEFAULT_FALLBACK_LANG
            instruct = instruct or DEFAULT_FALLBACK_INSTRUCT
        else:
            text = str(prompt)
            if not text:
                return DiffusionOutput(error="Empty text prompt")

        device = self.device
        num_cb = self.config.num_audio_codebook
        mask_id = self.config.audio_mask_id

        # 1. Encode reference audio tokens if provided (with voice caching) FIRST
        ref_audio_tokens = None
        if ref_audio is not None:
            if self.audio_tokenizer is None:
                raise RuntimeError(
                    "Voice cloning requires transformers>=5.3.0. Try: uv pip install 'transformers>=5.3.0'"
                )
            # Check speaker cache first
            _cache_key = None
            if voice_name:
                audio_hash = ""
                audio_sig_val = ref_audio[0] if isinstance(ref_audio, tuple) else ref_audio
                if audio_sig_val is not None:
                    import hashlib

                    raw_arr = audio_sig_val.cpu().numpy() if isinstance(audio_sig_val, torch.Tensor) else np.asarray(audio_sig_val)
                    audio_hash = hashlib.sha256(raw_arr.tobytes()[:8192]).hexdigest()[:8]

                _cache_key = self._speaker_cache.make_cache_key(
                    f"{voice_name}_{audio_hash}" if audio_hash else voice_name,
                    model_type="omnivoice",
                    created_at=int(prompt.get(OmniVoicePromptKey.VOICE_CREATED_AT) or 0) if isinstance(prompt, dict) else 0,
                )
                cached = self._speaker_cache.get(_cache_key)
                if cached is not None:
                    ref_audio_tokens = cached["ref_audio_tokens"].to(device)
                    _cache_key = None  # hit → don't store again
                    logger.debug("Speaker cache HIT for OmniVoice speaker '%s'", voice_name)

            if ref_audio_tokens is None:
                audio_signal, sr = ref_audio
                if isinstance(audio_signal, np.ndarray):
                    audio_signal = torch.from_numpy(audio_signal).float()
                ref_audio_tokens = self._encode_ref_audio(audio_signal, int(sr)).to(device)

                # Store in cache for next request
                if _cache_key is not None:
                    self._speaker_cache.put(_cache_key, {"ref_audio_tokens": ref_audio_tokens.cpu()})
                    logger.debug("Speaker cache STORE for OmniVoice speaker '%s'", voice_name)

        # 2. Universal dynamic target duration estimation and pacing normalization
        speed_val = (
            (prompt.get(OmniVoicePromptKey.SPEED) if isinstance(prompt, dict) else None)
            or extra.get(OmniVoicePromptKey.SPEED)
            or extra.get("speed")
            or mm_kwargs.get(OmniVoicePromptKey.SPEED)
            or DEFAULT_SPEED_FACTOR
        )
        try:
            speed_factor = float(speed_val)
        except (ValueError, TypeError):
            speed_factor = DEFAULT_SPEED_FACTOR

        if ref_text:
            ref_text = str(ref_text).strip()
            if ref_text and ref_text[-1] not in PUNCTUATION_SENTENCE_ENDERS:
                is_devanagari = any("\u0900" <= ch <= "\u097f" for ch in ref_text)
                ref_text = ref_text + ("।" if is_devanagari else ".")

        # Natural conversational pacing bounds (audio tokens per phonetic weight unit at 24kHz/960 hop):
        # Calibrated for natural, articulate conversational pacing (~14 - 15 characters/sec):
        # Baseline 1.60 tokens/weight yields brisk, energetic delivery. Range [1.40, 1.85] prevents sluggishness.
        BASELINE_TOKENS_PER_WEIGHT: float = 1.60
        MIN_TOKENS_PER_WEIGHT: float = 1.40
        MAX_TOKENS_PER_WEIGHT: float = 1.85
        TAIL_DECAY_TOKENS: int = 1  # ~40ms natural acoustic decay room

        target_weight = self.duration_estimator.calculate_total_weight(text)

        # Estimate reference pace ratio (tokens per weight unit)
        if ref_audio_tokens is not None and ref_text and len(str(ref_text).strip()) > 0:
            num_ref_tokens = ref_audio_tokens.shape[-1]
            ref_weight = self.duration_estimator.calculate_total_weight(str(ref_text).strip())
            if ref_weight > 0 and num_ref_tokens > 0:
                raw_ratio = num_ref_tokens / ref_weight
                # Smoothly normalize / clamp ratio into conversational bounds
                # If within bounds, blend gently with baseline to retain individual vocal tempo:
                clamped_ratio = max(MIN_TOKENS_PER_WEIGHT, min(MAX_TOKENS_PER_WEIGHT, raw_ratio))
                ratio = 0.65 * clamped_ratio + 0.35 * BASELINE_TOKENS_PER_WEIGHT
            else:
                ratio = BASELINE_TOKENS_PER_WEIGHT
        else:
            ratio = BASELINE_TOKENS_PER_WEIGHT

        # Base target duration in tokens
        target_len = int(round(target_weight * ratio)) + TAIL_DECAY_TOKENS

        # Dynamic user-configured speed factor (speed > 1.0 = faster, speed < 1.0 = slower)
        if speed_factor > 0 and speed_factor != DEFAULT_SPEED_FACTOR:
            target_len = int(round(target_len / speed_factor))
        target_len = max(1, target_len)

        # 3. Dynamic sampling parameters (num_step & guidance_scale)
        req_num_step = (
            (prompt.get(OmniVoicePromptKey.NUM_STEP) if isinstance(prompt, dict) else None)
            or extra.get(OmniVoicePromptKey.NUM_STEP)
            or self.num_step
        )
        req_guidance_scale = (
            (prompt.get(OmniVoicePromptKey.CFG_STRENGTH) if isinstance(prompt, dict) else None)
            or (prompt.get(OmniVoicePromptKey.GUIDANCE_SCALE) if isinstance(prompt, dict) else None)
            or extra.get(OmniVoicePromptKey.CFG_STRENGTH)
            or extra.get(OmniVoicePromptKey.GUIDANCE_SCALE)
            or self.guidance_scale
        )
        try:
            req_num_step = int(req_num_step)
        except (ValueError, TypeError):
            req_num_step = self.num_step
        try:
            req_guidance_scale = float(req_guidance_scale)
        except (ValueError, TypeError):
            req_guidance_scale = self.guidance_scale

        # 4. Build text prompt with control tokens
        style_text = ""
        if ref_audio_tokens is not None:
            style_text += OmniVoiceControlToken.DENOISE
        style_text += f"{OmniVoiceControlToken.LANG_START}{lang}{OmniVoiceControlToken.LANG_END}{OmniVoiceControlToken.INSTRUCT_START}{instruct}{OmniVoiceControlToken.INSTRUCT_END}"

        # Guard: Only condition on ref_text if ref_audio_tokens is actually present.
        # Without paired audio, prepending ref_text forces the model to synthesize ref_text aloud.
        effective_ref_text = ref_text if ref_audio_tokens is not None else None
        full_text = _combine_text(ref_text=effective_ref_text, text=text)
        wrapped_text = f"{OmniVoiceControlToken.TEXT_START}{full_text}{OmniVoiceControlToken.TEXT_END}"
        style_tokens = self.tokenizer.encode(style_text).ids
        text_tokens = _tokenize_with_nonverbal_tags(wrapped_text, self.tokenizer)
        encoding_ids = style_tokens + text_tokens
        text_tokens = torch.tensor(encoding_ids, dtype=torch.long, device=device)
        text_len = text_tokens.shape[0]

        # Build this request's conditional and unconditional token rows. They are
        # padded to a batch-wide length later, in _generate_batch.
        text_ids = text_tokens.unsqueeze(0).repeat(num_cb, 1)
        target_ids = torch.full((num_cb, target_len), mask_id, dtype=torch.long, device=device)

        if ref_audio_tokens is not None:
            cond_ids = torch.cat([text_ids, ref_audio_tokens, target_ids], dim=1)
        else:
            cond_ids = torch.cat([text_ids, target_ids], dim=1)

        return {
            "cond_ids": cond_ids,
            "uncond_ids": target_ids.clone(),
            "cond_len": cond_ids.shape[1],
            "uncond_len": target_len,
            "text_len": text_len,
            "target_len": target_len,
            "seed": seed,
            "num_step": req_num_step,
            "guidance_scale": req_guidance_scale,
        }

    def _generate_batch(self, items: list[dict]) -> list[torch.Tensor]:
        """Run the unmasking loop once for the whole batch, then decode each request."""
        device = self.device
        num_cb = self.config.num_audio_codebook
        mask_id = self.config.audio_mask_id
        n = len(items)

        # One padded length for the whole batch: the CFG halves of every request
        # must share a sequence dim to stack.
        max_len = max(max(it["cond_len"], it["uncond_len"]) for it in items)

        def _pad(ids: torch.Tensor) -> torch.Tensor:
            missing = max_len - ids.shape[1]
            if missing <= 0:
                return ids
            pad = torch.full((num_cb, missing), mask_id, dtype=torch.long, device=device)
            return torch.cat([ids, pad], dim=1)

        # Generator layout is [cond_0..cond_{B-1}, uncond_0..uncond_{B-1}]: it
        # reads the unconditional half of request i at row B+i.
        batch_input_ids = torch.stack([_pad(it["cond_ids"]) for it in items] + [_pad(it["uncond_ids"]) for it in items])

        batch_audio_mask = torch.zeros(2 * n, max_len, dtype=torch.bool, device=device)
        batch_attn_mask = torch.zeros(2 * n, 1, max_len, max_len, dtype=torch.bool, device=device)
        for i, it in enumerate(items):
            c_len, u_len = it["cond_len"], it["uncond_len"]
            batch_audio_mask[i, it["text_len"] : c_len] = True
            batch_audio_mask[n + i, :u_len] = True
            batch_attn_mask[i, :, :c_len, :c_len] = True
            batch_attn_mask[n + i, :, :u_len, :u_len] = True

        # A batch mixes requests whose seeds differ; the generator takes a single
        # seed, so a batched request is only bit-reproducible against another run
        # with the same batch composition. Requests carrying an explicit seed are
        # kept batch-1 upstream (see batch_compatibility_key) so their output
        # stays reproducible.
        seeds = [it["seed"] for it in items if it["seed"] is not None]
        seed = seeds[0] if len(seeds) == 1 and n == 1 else None

        batch_num_step = items[0].get(OmniVoicePromptKey.NUM_STEP, self.num_step) if items else self.num_step
        batch_guidance_scale = items[0].get(OmniVoicePromptKey.GUIDANCE_SCALE, self.guidance_scale) if items else self.guidance_scale

        target_lens = [it["target_len"] for it in items]
        tokens = self.generator(
            input_ids=batch_input_ids,
            audio_mask=batch_audio_mask,
            attention_mask=batch_attn_mask,
            target_lens=target_lens,
            num_step=batch_num_step,
            guidance_scale=batch_guidance_scale,
            t_shift=self.t_shift,
            layer_penalty_factor=self.layer_penalty_factor,
            position_temperature=self.position_temperature,
            class_temperature=self.class_temperature,
            seed=seed,
        )  # [B, 8, max_target_len]

        # Decode per request: tokens are padded to the batch's longest target, so
        # decoding the batch whole would run the codec over another request's
        # padding and stretch this request's audio.
        return [self.decoder(tokens[i : i + 1, :, : target_lens[i]]) for i in range(n)]

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights from model directory (not from the iterator).

        The diffusion model loader passes HF safetensors weights, but OmniVoice
        has custom weight names (llm.* → generator.*, audio_tokenizer.* → decoder.*).
        We load from model_path directly and return all param names to satisfy
        the loader's "all weights initialized" check.
        """
        # Consume the iterator (required by the loader contract)
        for _ in weights:
            pass

        device = self.device
        self.generator.load_weights(self.model_path, device)
        self.generator = self.generator.to(device).eval()
        self.decoder.load_weights(self.model_path, device)
        logger.info("OmniVoice pipeline loaded on %s", device)

        # Return all parameter names to indicate they're initialized
        return {name for name, _ in self.named_parameters()}
