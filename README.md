```bash
# Train
$ CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run python -OO ./LunarLander/policy-gradient/main.py train --env LunarLander-v3 --save_to ./.scratch/model_$(date +%s).pt

# Infer
$ CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run python -OO ./LunarLander/policy-gradient/main.py infer --load_from ./.scratch/model_1727759757.pt
```
