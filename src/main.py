import importlib.util
import sys
from pathlib import Path

# Locate workflow file (supports both 'agentic' and 'com.agentic' directories)
_src_dir = Path(__file__).resolve().parent
_workflow_path = _src_dir / "agentic" / "box-agentic-workflow.py"

if not _workflow_path.is_file():
    _workflow_path = _src_dir / "com.agentic" / "box-agentic-workflow.py"

if not _workflow_path.is_file():
    raise FileNotFoundError(
        f"Could not locate 'box-agentic-workflow.py' in '{_src_dir / 'agentic'}' "
        f"or '{_src_dir / 'com.agentic'}'"
    )

_spec = importlib.util.spec_from_file_location("box_agentic_workflow", _workflow_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load module from {_workflow_path}")

_module = importlib.util.module_from_spec(_spec)
# Must register in sys.modules before exec_module for Pydantic/FastAPI introspection
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)

# Expose FastAPI instance for Uvicorn
app = _module.app

__all__ = ["app"]