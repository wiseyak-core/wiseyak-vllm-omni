# wiseyak fork — deployment and upstream workflow

This is `wiseyak-core/wiseyak-vllm-omni`, a fork of `vllm-project/vllm-omni` carrying
wiseyak's OmniVoice optimizations and publishing the image that wiseai-tts serves
Nepali TTS from.

wiseyak-only file — never include it in an upstream PR.

## Remotes

```
upstream  https://github.com/vllm-project/vllm-omni.git       (read-only)
origin    https://github.com/wiseyak-core/wiseyak-vllm-omni.git
```

`main` is a read-only mirror of `upstream/main`. Never commit to it.

## The branch contract

`wiseyak-omnivoice` = an upstream release tag + a short, ordered patch series in two
groups. Keeping the groups separate is what makes rebases mechanical and upstream PRs
cheap to cut:

1. **`opt:` commits** — OmniVoice model and pipeline optimizations: fp16 serving, the
   fused QKV+RoPE hot loop, request-level batching, and the `OMNIVOICE_NUM_STEP` /
   `OMNIVOICE_CUDA_GRAPH_BATCH_SIZES` knobs. Touch only `vllm_omni/` and `tests/`.
   These are the upstream-PR-able ones and must stay free of any packaging file.
2. **One `deploy:` commit, always last** — `Jenkinsfile`, `docker/Dockerfile.wiseyak`,
   `.dockerignore`, this file. Never sent upstream.

Check the split holds before any rebase or PR:

```bash
git log --oneline wiseyak/base-v0.28.0..wiseyak-omnivoice
git diff --stat upstream/main...wiseyak-omnivoice
```

## Rebasing onto a new upstream release

Record the base each time as a **branch**, `wiseyak/base-<upstream-tag>`, so `--onto` is
never a guess:

```bash
git fetch upstream --tags
git branch wiseyak/base-v0.29.0 v0.29.0

git checkout wiseyak-omnivoice
git rebase --onto v0.29.0 wiseyak/base-v0.28.0
```

**A branch, never a tag.** `setuptools-scm` derives the package version from
`git describe --tags`, so any tag placed on this branch becomes the nearest tag and is
parsed as the version. `wiseyak/base-v0.28.0` is not a valid version string, and the build
fails inside `uv pip install .` with `InvalidVersion: Invalid version: 'dev'`. Branches are
invisible to `git describe --tags`, so they mark the base without touching versioning. Keep
`git describe --tags` reading `v0.28.0-84-g0eaab404` — an upstream tag plus a distance.

Then, before pushing:

1. Bump the `FROM vllm/vllm-openai:vX.Y.Z` pin in `docker/Dockerfile.wiseyak` to match
   the new base, **in the same commit as the rebase**. A mismatch between the base
   image's vLLM and this checkout's expected APIs is exactly what broke the previous
   0.26-based overlay image.
2. Re-sync `docker/Dockerfile.wiseyak` against upstream's `docker/Dockerfile.cuda` — it
   is a copy of that plus one `/opt/omnivoice/deploy.yaml` layer, so upstream changes to
   the install steps need porting.
3. `pytest tests/diffusion/models/omnivoice tests/model_executor/models/omnivoice`
4. Drop any `opt:` commit that has since been merged upstream — the series should shrink
   over time.

```bash
git push --force-with-lease origin wiseyak-omnivoice
git push origin wiseyak/base-v0.29.0
```

Always `--force-with-lease`, never bare `-f`: the lease is what stops you overwriting a
push someone else made to the shared branch.

## Sending a PR upstream

The fork's rename is not an obstacle — GitHub keys the fork relationship off the
repository ID, not the name, so cross-fork PRs work normally; the head ref just reads
`wiseyak-core:<branch>`.

What *is* an obstacle is opening the PR from `wiseyak-omnivoice` directly, because its tip
carries the `deploy:` commit. Cut a clean topic branch off upstream instead:

```bash
git fetch upstream
git checkout -b upstream/omnivoice-fp16-batching upstream/main
git cherry-pick <the opt: commits only>
git push origin upstream/omnivoice-fp16-batching

gh pr create --repo vllm-project/vllm-omni \
  --base main --head wiseyak-core:upstream/omnivoice-fp16-batching \
  --title "[Perf][OmniVoice] fp16 serving and request-level batching"
```

For the PR body use the measurements already in the header of
`vllm_omni/deploy/omnivoice.yaml` (RTX 4090, 12-sentence Whisper-small gate: 2.60 → 8.63
req/s, RTF 0.090 → 0.027, WER 0.0641 unchanged) — a perf PR will be asked for exactly
that. Run the repo's `precheck-pr` skill first, and update the stale "one float32
diffusion stage" line in `recipes/k2-fsa/OmniVoice.md`.

## The image

`docker/Dockerfile.wiseyak` is built, pushed and verified by `deploy.sh`, which lives
**outside this repo** (in `vllm-omni-deploy/` next to this checkout, together with its
Jenkinsfile) so it survives rebases and never enters a diff against upstream:

```
registry.wiseai.wiseyak.com/wiseyak-vllm-omni:<sha>     ./deploy.sh build push
registry.wiseai.wiseyak.com/wiseyak-vllm-omni:latest    ./deploy.sh release  (only after verify)
```

The `<sha>` tag is what makes rollback a one-liner — see `OMNIVOICE_IMAGE_TAG` in
wiseai-tts's `deployment/docker_compose/docker-compose.omnivoice-optimized.yml`. For a
readable tag on a validated release: `RELEASE_TAG=v0.28.0-wiseyak.1 ./deploy.sh release`.

`deploy.sh build` refuses to build unless this `deploy:` commit is HEAD, the branch is
pushed, and the `FROM` pin matches the upstream tag from `git describe --tags`.

Model weights are never baked in — wiseai-tts bind-mounts them at `/app/models`.
