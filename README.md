See `pyproject.toml` for `[project.scripts]`. Run using:

```bash
# Train
$ CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run $SCRIPT train --env $ENV_ID --save_to ./.scratch/model_$(date +%s).pt

# Infer
$ CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run $SCRIPT infer --load_from ./.scratch/model_1727759757.pt
```
