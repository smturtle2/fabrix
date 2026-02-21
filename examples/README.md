## Fabrix Examples

### Minimal

- `examples/minimal/quickstart.py`
  - Smallest tool-calling flow (`reasoning -> tool_call -> response`).
- `examples/minimal/multimodal.py`
  - Image inputs via URL, local path, bytes, and two-image comparison.

### Advanced

- `examples/advanced/data_workflow.py`
  - Deterministic 3-step data pipeline orchestration.
- `examples/advanced/incident_response.py`
  - Multi-tool incident triage and external update drafting.
- `examples/advanced/complex_reasoning_workflow.py`
  - Multi-constraint decision task where consecutive reasoning iterations tend to
    emerge naturally from trade-off analysis and stress-test validation.

### Run

From repository root:

```bash
PYTHONPATH=src python3 examples/minimal/quickstart.py
PYTHONPATH=src python3 examples/minimal/multimodal.py
PYTHONPATH=src python3 examples/advanced/data_workflow.py
PYTHONPATH=src python3 examples/advanced/incident_response.py
PYTHONPATH=src python3 examples/advanced/complex_reasoning_workflow.py
```
