#!/bin/bash
set -e

VERL_FILE=$(python - <<'PY'
import importlib.util
from pathlib import Path

spec = importlib.util.find_spec("verl")
if spec is None or spec.origin is None:
    raise RuntimeError("Could not find installed verl package.")

verl_init = Path(spec.origin)
verl_dir = verl_init.parent
fsdp_file = verl_dir / "workers" / "fsdp_workers.py"

print(fsdp_file)
PY
)

echo "Patching verl file:"
echo "$VERL_FILE"

if [ ! -f "$VERL_FILE" ]; then
    echo "File not found: $VERL_FILE"
    exit 1
fi

if grep -q "if role == 'actor' and optim_config is not None:" "$VERL_FILE"; then
    echo "Patch already applied."
    exit 0
fi

if grep -q "if role == 'actor':" "$VERL_FILE"; then
    sed -i "s/if role == 'actor':/if role == 'actor' and optim_config is not None:/" "$VERL_FILE"
    echo "Patch applied successfully."
else
    echo "Expected line not found. Please inspect $VERL_FILE manually."
    exit 1
fi
