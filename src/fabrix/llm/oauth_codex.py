from __future__ import annotations

import copy
import json
from typing import Any

from oauth_codex import OAuthCodexClient
from oauth_codex.tooling import build_strict_response_format

from fabrix.errors import LLMOutputError
from fabrix.graph.state import NextState, StateEnvelope
from fabrix.graph.transitions import allowed_next_states
from fabrix.types import ReasoningEffort

_DISALLOWED_SCHEMA_KEYWORDS = {
    "oneOf",
    "allOf",
    "not",
    "if",
    "then",
    "else",
    "patternProperties",
    "const",
    "$ref",
    "$defs",
    "definitions",
}


class OAuthCodexStateProvider:
    def __init__(
        self,
        *,
        instructions: str,
        client: OAuthCodexClient | None = None,
        model: str = "gpt-5.3-codex",
        reasoning_effort: ReasoningEffort = "medium",
    ) -> None:
        self._client = client or OAuthCodexClient()
        self._instructions = instructions
        self._model = model
        self._reasoning_effort = reasoning_effort

    async def generate_state(
        self,
        *,
        task: str,
        context: dict[str, Any],
        history: list[dict[str, Any]],
        current_state: NextState,
        step: int,
        tool_schemas: list[dict[str, Any]],
    ) -> StateEnvelope:
        prompt = self._build_prompt(
            task=task,
            context=context,
            history=history,
            current_state=current_state,
            step=step,
            tool_schemas=tool_schemas,
        )
        output_schema = self._build_output_schema(
            current_state=current_state,
            tool_schemas=tool_schemas,
        )

        try:
            payload = await self._client.agenerate(
                prompt=prompt,
                model=self._model,
                reasoning_effort=self._reasoning_effort,
                output_schema=output_schema,
                strict_output=True,
            )
        except Exception as exc:
            raise LLMOutputError(f"failed to produce valid state output: {exc}") from exc

        return self._parse_payload(payload)

    def _parse_payload(self, payload: str | dict[str, Any]) -> StateEnvelope:
        if isinstance(payload, str):
            try:
                data = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise LLMOutputError("model returned non-JSON text for structured state") from exc
        else:
            data = payload
        return StateEnvelope.model_validate(data)

    def _build_output_schema(
        self,
        *,
        current_state: NextState,
        tool_schemas: list[dict[str, Any]],
    ) -> dict[str, Any]:
        state_schema = self._build_state_schema(
            current_state=current_state,
            tool_schemas=tool_schemas,
        )
        schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "StateEnvelope",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"state": state_schema},
                    "required": ["state"],
                    "additionalProperties": False,
                },
            },
        }
        return self._normalize_and_validate_schema(schema)

    def _build_state_schema(
        self,
        *,
        current_state: NextState,
        tool_schemas: list[dict[str, Any]],
    ) -> dict[str, Any]:
        allowed_next = self._allowed_next_state_values(
            current_state=current_state,
            tool_schemas=tool_schemas,
        )
        next_state_schema = {"type": "string", "enum": allowed_next}

        if current_state is NextState.reasoning:
            return {
                "type": "object",
                "properties": {
                    "state_type": {"type": "string", "enum": ["reasoning"]},
                    "next_state": next_state_schema,
                    "reasoning": {"type": "string"},
                    "focus": {"type": "string"},
                },
                "required": ["state_type", "next_state", "reasoning", "focus"],
                "additionalProperties": False,
            }

        if current_state is NextState.tool_call:
            if not tool_schemas:
                raise LLMOutputError("tool_call state requested but no tools are registered")
            return {
                "type": "object",
                "properties": {
                    "state_type": {"type": "string", "enum": ["tool_call"]},
                    "next_state": next_state_schema,
                    "tool_calls": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"anyOf": [self._build_tool_call_item_schema(s) for s in tool_schemas]},
                    },
                },
                "required": ["state_type", "next_state", "tool_calls"],
                "additionalProperties": False,
            }

        if current_state is NextState.response:
            return {
                "type": "object",
                "properties": {
                    "state_type": {"type": "string", "enum": ["response"]},
                    "next_state": next_state_schema,
                    "response": {"type": "string"},
                    "audience": {"type": "string", "enum": ["user", "system"]},
                },
                "required": ["state_type", "next_state", "response", "audience"],
                "additionalProperties": False,
            }

        if current_state is NextState.finish:
            return {
                "type": "object",
                "properties": {
                    "state_type": {"type": "string", "enum": ["finish"]},
                    "next_state": next_state_schema,
                    "final_output": {"type": "string"},
                    "completion_reason": {"type": "string"},
                },
                "required": ["state_type", "next_state", "final_output", "completion_reason"],
                "additionalProperties": False,
            }

        raise LLMOutputError(f"unsupported current state: {current_state}")

    def _build_tool_call_item_schema(self, tool_schema: dict[str, Any]) -> dict[str, Any]:
        name = tool_schema.get("name")
        if not isinstance(name, str) or not name:
            raise LLMOutputError("tool schema has invalid or missing `name`")

        parameters = tool_schema.get("parameters")
        if not isinstance(parameters, dict):
            raise LLMOutputError(f"tool `{name}` has invalid `parameters` schema")

        return {
            "type": "object",
            "properties": {
                "name": {"type": "string", "enum": [name]},
                "arguments": self._strictify_parameters_schema(parameters),
            },
            "required": ["name", "arguments"],
            "additionalProperties": False,
        }

    def _strictify_parameters_schema(self, schema: dict[str, Any]) -> dict[str, Any]:
        if schema.get("type") != "object":
            raise LLMOutputError("tool parameter schema root must be an object")

        try:
            strict_format = build_strict_response_format(
                {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "ToolArguments",
                        "schema": copy.deepcopy(schema),
                    },
                }
            )
        except Exception as exc:
            raise LLMOutputError(f"failed to strictify tool parameter schema: {exc}") from exc

        strict_schema = strict_format.get("schema")
        if not isinstance(strict_schema, dict):
            raise LLMOutputError("strictified tool parameter schema must be a dict")
        if strict_schema.get("type") != "object":
            raise LLMOutputError("strictified tool parameter schema root must be an object")
        if strict_schema.get("additionalProperties") is not False:
            raise LLMOutputError(
                "strictified tool parameter schema must set additionalProperties=false"
            )

        return self._normalize_and_validate_schema(self._inline_local_refs(strict_schema))

    def _inline_local_refs(self, schema: dict[str, Any]) -> dict[str, Any]:
        root = copy.deepcopy(schema)

        def resolve_pointer(pointer: str) -> dict[str, Any]:
            if not pointer.startswith("#/"):
                raise LLMOutputError(f"unsupported $ref pointer: {pointer}")

            target: Any = root
            for raw_part in pointer[2:].split("/"):
                part = raw_part.replace("~1", "/").replace("~0", "~")
                if not isinstance(target, dict) or part not in target:
                    raise LLMOutputError(f"unresolvable $ref pointer: {pointer}")
                target = target[part]

            if not isinstance(target, dict):
                raise LLMOutputError(f"$ref pointer does not resolve to object: {pointer}")
            return copy.deepcopy(target)

        def walk(node: Any) -> Any:
            if isinstance(node, dict):
                ref = node.get("$ref")
                if isinstance(ref, str):
                    merged = resolve_pointer(ref)
                    for key, value in node.items():
                        if key == "$ref":
                            continue
                        merged[key] = walk(value)
                    return walk(merged)

                out: dict[str, Any] = {}
                for key, value in node.items():
                    if key in {"$defs", "definitions"}:
                        continue
                    out[key] = walk(value)
                return out

            if isinstance(node, list):
                return [walk(item) for item in node]

            return node

        result = walk(root)
        if not isinstance(result, dict):
            raise LLMOutputError("inlined schema must be an object")
        return result

    def _normalize_and_validate_schema(self, schema: Any) -> dict[str, Any]:
        normalized = self._normalize_schema_subset(schema)
        self._assert_no_disallowed_keywords(normalized)
        self._assert_object_required_consistency(normalized)
        if not isinstance(normalized, dict):
            raise LLMOutputError("normalized schema must be an object")
        return normalized

    def _normalize_schema_subset(self, schema: Any) -> Any:
        if isinstance(schema, dict):
            normalized: dict[str, Any] = {}
            for key, value in schema.items():
                normalized_key = "anyOf" if key == "oneOf" else key
                if normalized_key == "const":
                    continue
                normalized[normalized_key] = self._normalize_schema_subset(value)

            if "const" in schema and "enum" not in normalized:
                normalized["enum"] = [schema["const"]]

            if normalized.get("type") == "object":
                properties = normalized.get("properties")
                if not isinstance(properties, dict):
                    properties = {}
                normalized["properties"] = properties
                normalized["required"] = list(properties.keys())
                normalized["additionalProperties"] = False

            return normalized

        if isinstance(schema, list):
            return [self._normalize_schema_subset(item) for item in schema]

        return schema

    def _assert_no_disallowed_keywords(self, schema: Any) -> None:
        if isinstance(schema, dict):
            for key, value in schema.items():
                if key in _DISALLOWED_SCHEMA_KEYWORDS:
                    raise LLMOutputError(f"schema contains disallowed keyword: {key}")
                self._assert_no_disallowed_keywords(value)
            return

        if isinstance(schema, list):
            for item in schema:
                self._assert_no_disallowed_keywords(item)

    def _assert_object_required_consistency(self, schema: Any, path: str = "$") -> None:
        if isinstance(schema, dict):
            if schema.get("type") == "object":
                properties = schema.get("properties")
                required = schema.get("required")
                if not isinstance(properties, dict):
                    raise LLMOutputError(f"schema object missing properties at {path}")
                if not isinstance(required, list):
                    raise LLMOutputError(f"schema object missing required list at {path}")
                property_keys = list(properties.keys())
                if required != property_keys:
                    raise LLMOutputError(
                        "schema object required/properties mismatch at "
                        f"{path}: required={required}, properties={property_keys}"
                    )

            for key, value in schema.items():
                self._assert_object_required_consistency(value, f"{path}.{key}")
            return

        if isinstance(schema, list):
            for index, item in enumerate(schema):
                self._assert_object_required_consistency(item, f"{path}[{index}]")

    def _allowed_next_state_values(
        self,
        *,
        current_state: NextState,
        tool_schemas: list[dict[str, Any]],
    ) -> list[str]:
        allowed = list(allowed_next_states(current_state))
        if not tool_schemas and NextState.tool_call in allowed:
            allowed = [state for state in allowed if state is not NextState.tool_call]

        values = [state.value for state in allowed]
        if not values:
            raise LLMOutputError(f"no allowed next_state available from {current_state.value}")
        return values

    def _build_prompt(
        self,
        *,
        task: str,
        context: dict[str, Any],
        history: list[dict[str, Any]],
        current_state: NextState,
        step: int,
        tool_schemas: list[dict[str, Any]],
    ) -> str:
        allowed = self._allowed_next_state_values(
            current_state=current_state,
            tool_schemas=tool_schemas,
        )
        no_tools_line = (
            "No tools are registered. Never choose next_state=tool_call.\n"
            if not tool_schemas
            else ""
        )

        return (
            "You are Fabrix, a graph-based agent state generator.\n"
            "Return ONLY a JSON object matching the provided schema.\n"
            f"Current node/state_type MUST be `{current_state.value}`.\n"
            f"Allowed next_state values now: {allowed}.\n"
            "Graph rules:\n"
            "- reasoning -> reasoning|tool_call|response|finish\n"
            "- tool_call -> reasoning\n"
            "- response -> reasoning|finish\n"
            "- finish -> finish\n"
            "Tool usage rules:\n"
            "- Use tool_call state only when external computation/data access is needed.\n"
            "- In tool_call state, each arguments object must exactly match selected tool schema.\n"
            "- In tool_call state, include one or more tool_calls.\n"
            f"{no_tools_line}"
            "Response rules:\n"
            "- response state is an intermediate user-facing reply.\n"
            "- finish state must include final_output and completion_reason.\n"
            "\n"
            "Developer instructions:\n"
            f"{self._instructions}\n"
            "\n"
            f"Step: {step}\n"
            f"Task: {task}\n"
            f"Context JSON: {json.dumps(context, ensure_ascii=True)}\n"
            f"Available tools JSON schema: {json.dumps(tool_schemas, ensure_ascii=True)}\n"
            f"Execution history JSON: {json.dumps(history[-20:], ensure_ascii=True)}\n"
        )
