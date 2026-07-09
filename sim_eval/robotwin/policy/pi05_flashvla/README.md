# pi05_flashvla — RoboTwin adapter for FlashVLA PI05 FlashVLA

Runs a flashvla `PI05FlashVLAPolicy` checkpoint against RoboTwin tasks
using RoboTwin's **client/server** eval architecture.

## Why client/server?

The flashvla training environment and the RoboTwin SAPIEN environment have
**hard version conflicts** — they cannot coexist in a single conda env:

| Package | flashvla env | RoboTwin env | Why it matters |
|---|---|---|---|
| torch        | **2.9.1+cu128** | **2.4.1** (pinned by sapien/mplib C++ ext) | Different ABIs |
| numpy        | **2.x** | **1.x** (sapien/open3d/pytorch3d C ext)    | ABI break at numpy 2.0 |
| gymnasium    | **1.2.x** | **0.29.x** (breaking API change)          | Different Env interface |
| python       | 3.12            | 3.10 (sapien 3.0.0b1 wheel limit)          | ABI |
| sapien/mplib | — not installed | required                                   | C++ ext → specific env |
| flashvla/lerobot| editable install| — not needed                                | heavy deps |

Trying to stuff both into one env breaks one or the other. RoboTwin ships
a `policy_model_server.py` + `eval_policy_client.py` pair exactly for this
case: the neural model runs in its own process (and env), the simulator
runs in another process (and env), and they talk via a local TCP socket
with a tiny JSON+base64 protocol.

## Files

```
pi05_flashvla/
├── __init__.py              # from .deploy_policy import *
├── deploy_policy.py         # LIGHT: get_model lazy-imports flashvla; eval() uses model.call()
├── deploy_policy.yml        # shared config (unused fields ignored by each side)
├── pi05_flashvla_model.py   # HEAVY: imports flashvla → defines PI05FlashVLAModel (server only)
├── eval_server.sh                 # start model server (flashvla env)
├── eval_client.sh           # run eval against server (RoboTwin env)
└── README.md                # this file
```

## Setup

### 1. Ensure flashvla is installed in the flashvla env (already done)

```bash
conda activate flashvla
python -c "from flashvla.policies.pi05.modeling_pi05_flashvla import PI05FlashVLAPolicy; print('ok')"
```

### 2. Create a dedicated RoboTwin conda env (following RoboTwin's own install guide)

```bash
conda create -n robotwin python=3.10 -y
conda activate robotwin
cd /path/to/RoboTwin
bash script/_install.sh
# This installs: torch==2.4.1, sapien==3.0.0b1, mplib==0.2.1,
#   gymnasium==0.29.1, transforms3d, trimesh, open3d, curobo, pytorch3d, ...
# Then download RoboTwin assets per their README.
```

Do **NOT** `pip install -e /path/to/flashvla` into the robotwin env —
it would try to pull in the FlashVLA torch stack and break SAPIEN.

## Running an evaluation

You need **two terminals**, each in a different conda env.

### Terminal 1 — model server (flashvla env)

```bash
conda activate flashvla
cd /path/to/RoboTwin/policy/pi05_flashvla
bash eval_server.sh \
    /home/zekail/runs/flashvla_action_robotwin_multitask/checkpoints/last/pretrained_model \
    9999 0
```

Expected output:
```
[pi05_flashvla] loaded /home/zekail/runs/.../pretrained_model
[pi05_flashvla] cold_start_mode=current_state, device=cuda:0
🚀 Model server started on localhost:9999
🔄 Server is waiting for client connections...
```

### Terminal 2 — eval client (robotwin env)

```bash
conda activate robotwin
cd /path/to/RoboTwin/policy/pi05_flashvla
bash eval_client.sh beat_block_hammer demo_clean 9999 0 0
```

The client runs RoboTwin's standard 100-episode sweep for that task, with
the network latency between the two envs being negligible compared to
SAPIEN simulation time. The per-task result file is written to
`RoboTwin/eval_result/<task>/pi05_flashvla/<config>/.../`.

### Sweep multiple tasks

Keep the server terminal running and just re-run `eval_client.sh` with
different `task_name` / `task_config`. The server is stateless across
episodes (it resets between episodes via `model.call('reset_model')`).

```bash
# In Terminal 2
for task in beat_block_hammer click_bell stack_blocks_two; do
    bash eval_client.sh $task demo_randomized 9999 0 0
done
```

## Protocol details

- `eval_server.sh` → `RoboTwin/script/policy_model_server.py` →
  `deploy_policy.get_model(usr_args)` → `PI05FlashVLAModel(...)` listening
  on port 9999. Server blocks, handles each client in a daemon thread.
- `eval_client.sh` → `RoboTwin/script/eval_policy_client.py`:
  - Loads the RoboTwin task env (SAPIEN), does the expert sanity check.
  - Creates `model = ModelClient(port=9999)` instead of calling `get_model`.
  - Before each episode: `model.call(func_name='reset_model')`.
  - Per env step:
      * `eval_func(TASK_ENV, model, observation)` from `deploy_policy.eval`
      * which calls `model.call(func_name='get_action', obs={head_rgb, left_rgb, right_rgb, state, instruction})`
      * server unpacks → runs pi05 FlashVLA → returns 14-dim qpos
      * client passes it to `TASK_ENV.take_action(action)`

## Cold start

Action streaming maintains a buffer of N=5 slots with different noise
levels. The first `(N-1)=4` chunks (≈40 env steps) happen while the
buffer is still filling — during this phase `PI05FlashVLAModel.get_action`
returns the current joint position so the arm holds its pose.
`cold_start_mode=current_state` is set at load time regardless of what
the checkpoint was trained with (RoboTwin actions are absolute qpos, so
`zero_delta` would send the arm to joint zero and crash).

Once the buffer fills, every subsequent call runs 1 denoising step,
extracts the cleanest slot, shifts the buffer, and appends fresh noise —
producing a new chunk every observation.
