# Real-Robot Deployment

For real robots, FlashVLA provides exactly one integration point:
[`flashvla.async_manager.AsyncStreamingActionManager`](../../flashvla/async_manager.py).
It wraps a FlashVLA policy with async chunk-overlap execution and the
streaming cold-start fallback; you bring the robot driver and the control
loop.

## The contract

```python
from flashvla.async_manager import AsyncStreamingActionManager

mgr = AsyncStreamingActionManager(policy, overlap_steps=1)
mgr.warmup(preprocessor(first_obs))   # capture CUDA graph off-robot (10-30s once)
mgr.reset()                           # pristine buffer at episode start
while running:
    action = mgr.act(preprocessor(obs))   # one action per call, [B, action_dim]
    robot.send_action(postprocessor(action)[0].cpu().numpy())
```

- `overlap_steps=N` launches the next chunk inference N control steps before
  the current chunk runs out; under `compile_model=true` the launch is
  dispatch-only and the GPU work hides behind robot execution. `0` = sync.
- During the streaming cold start (first `num_buffer_slots - 1` chunks) the
  manager emits a hold-still action derived from the checkpoint's
  normalization stats:
  - `cold_start_mode: zero_delta` — unnormalizes to a zero action
    (delta-action robots).
  - `cold_start_mode: current_state` — unnormalizes to the current qpos
    (absolute-position robots; **required** if a zero command would move the
    arm).
- Call `mgr.reset()` at every episode boundary — it clears the manager AND
  the policy's rolling denoise buffer.

## Practical notes

- Pick `overlap_steps` so that one chunk inference fits inside
  `overlap_steps / fps` seconds; `1` is the safe default (LIBERO/RoboTwin
  sweeps showed `o=1` ties or beats sync almost everywhere, larger overlaps
  trade accuracy for staleness).
- Observation keys and camera names must exactly match the training dataset
  features (`observation.images.<cam>`, `observation.state`).
- The processors saved with the checkpoint (`policy_preprocessor.json` /
  `policy_postprocessor.json`) handle normalization, prompt construction and
  tokenization — always load them via
  `flashvla.policies.factory.make_pre_post_processors(cfg, pretrained_path=ckpt)`.
