import importlib.util
from pathlib import Path

# Dynamically import app from box-agentic-workflow.py despite non-standard characters ('.' and '-')
_workflow_path = (
    Path(__file__).resolve().parent / "com.agentic" / "box-agentic-workflow.py"
)

_spec = importlib.util.spec_from_file_location("box_agentic_workflow", _workflow_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load module from {_workflow_path}")

_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

# Expose FastAPI instance for Uvicorn
app = _module.app