from __future__ import annotations

import copy
import json
from datetime import date, datetime, time
from enum import Enum
from typing import Any

from oauth_codex import OAuthCodexClient
from oauth_codex.tooling import build_strict_response_format
from pydantic import BaseModel

from fabrix.errors import LLMOutputError
from fabrix.graph.state import (
    FinishState,
    NextState,
    ReasoningState,
    ResponseState,
    StateEnvelope,
    ToolCallState,
)
from fabrix.graph.transitions import allowed_next_states
from fabrix.messages import ImageMessage, TextMessage, to_oauth_message
from fabrix.types import ReasoningEffort

DEFAULT_MODEL = "gpt-5.3-codex"

_UNSUPPORTED_SCHEMA_KEYWORDS = {
    "$ref",
    "$defs",
    "definitions",
    "oneOf",
    "const",
}

_STATE_MODEL_BY_STATE: dict[
    NextState,
    type[ReasoningState | ToolCallState | ResponseState | FinishState],
] = {
    NextState.reasoning: ReasoningState,
    NextState.tool_call: ToolCallState,
    NextState.response: ResponseState,
    NextState.finish: FinishState,
}


class OAuthCodexStateProvider:
    def __init__(
        self,
        *,
        instructions: str,
        client: OAuthCodexClient | None = None,
        model: str = DEFAULT_MODEL,
        reasoning_effort: ReasoningEffort = "medium",
    ) -> None:
        self._client = client or OAuthCodexClient()
        self._instructions = instructions
        self._model = model
        self._reasoning_effort = reasoning_effort

    def validate_tool_schemas(self, tool_schemas: list[dict[str, Any]]) -> None:
        if tool_schemas:
            self._build_output_schema(
                current_state=NextState.tool_call,
                tool_schemas=tool_schemas,
            )

        for state in (NextState.reasoning, NextState.response, NextState.finish):
            self._build_output_schema(
                current_state=state,
                tool_schemas=tool_schemas,
            )

    async def generate_state(
        self,
        *,
        messages: list[TextMessage | ImageMessage],
        history: list[dict[str, Any]],
        current_state: NextState,
        step: int,
        tool_schemas: list[dict[str, Any]],
    ) -> StateEnvelope:
        prompt = self._build_prompt(
            messages=messages,
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
                messages=self._build_model_messages(prompt=prompt, messages=messages),
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
        output_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": StateEnvelope.__name__,
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"state": state_schema},
                    "required": ["state"],
                    "additionalProperties": False,
                },
            },
        }
        return self._normalize_schema(output_schema)

    def _build_state_schema(
        self,
        *,
        current_state: NextState,
        tool_schemas: list[dict[str, Any]],
    ) -> dict[str, Any]:
        model_type = _STATE_MODEL_BY_STATE.get(current_state)
        if model_type is None:
            raise LLMOutputError(f"unsupported current state: {current_state}")

        strict_format = build_strict_response_format(model_type)
        schema = strict_format.get("schema")
        if not isinstance(schema, dict):
            raise LLMOutputError(f"strict state schema for `{current_state.value}` must be a dict")

        state_schema = copy.deepcopy(self._normalize_schema(schema))
        properties = state_schema.get("properties")
        if not isinstance(properties, dict):
            raise LLMOutputError(f"state schema for `{current_state.value}` missing properties")

        next_state_schema = properties.get("next_state")
        if not isinstance(next_state_schema, dict):
            raise LLMOutputError(f"state schema for `{current_state.value}` missing next_state")
        next_state_schema["enum"] = self._allowed_next_state_values(
            current_state=current_state,
            tool_schemas=tool_schemas,
        )

        if current_state is NextState.tool_call:
            if not tool_schemas:
                raise LLMOutputError("tool_call state requested but no tools are registered")
            tool_calls_schema = properties.get("tool_calls")
            if not isinstance(tool_calls_schema, dict):
                raise LLMOutputError("state schema for `tool_call` missing tool_calls")
            tool_calls_schema["items"] = {
                "anyOf": [self._build_tool_call_item_schema(schema) for schema in tool_schemas]
            }

        return self._normalize_schema(state_schema)

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
                "arguments": self._strictify_parameters_schema(parameters, tool_name=name),
            },
            "required": ["name", "arguments"],
            "additionalProperties": False,
        }

    def _strictify_parameters_schema(
        self,
        schema: dict[str, Any],
        *,
        tool_name: str,
    ) -> dict[str, Any]:
        if schema.get("type") != "object":
            raise LLMOutputError("tool parameter schema root must be an object")

        schema_name = f"{tool_name}Arguments"

        try:
            strict_format = build_strict_response_format(
                {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "schema": copy.deepcopy(schema),
                    },
                }
            )
        except Exception as exc:
            raise LLMOutputError(f"failed to strictify tool parameter schema: {exc}") from exc

        strict_schema = strict_format.get("schema")
        if not isinstance(strict_schema, dict):
            raise LLMOutputError("strictified tool parameter schema must be a dict")

        return self._normalize_schema(strict_schema)

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

    def _render_graph_rules(self) -> str:
        lines: list[str] = []
        for state in NextState:
            allowed = "|".join(next_state.value for next_state in allowed_next_states(state))
            lines.append(f"- {state.value} -> {allowed}")
        return "\n".join(lines)

    def _build_prompt(
        self,
        *,
        messages: list[TextMessage | ImageMessage],
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
            f"{self._render_graph_rules()}\n"
            "Tool usage rules:\n"
            "- Use tool_call state only when external computation/data access is needed.\n"
            "- In tool_call state, each arguments object must exactly match selected tool schema.\n"
            "- In tool_call state, include one or more tool_calls.\n"
            f"{no_tools_line}"
            "Response rules:\n"
            "- response state is an intermediate user-facing reply.\n"
            "- finish state must include final_output and completion_reason.\n"
            "Reasoning loop strategy:\n"
            "- Use Chain-of-Thought-style multi-step planning through short, visible decision traces.\n"
            "- Keep each reasoning step to 1-2 sentences with one concrete focus.\n"
            "- If uncertainty remains, choose next_state=reasoning; usually resolve within several reasoning steps.\n"
            "- Each step must add new evidence or a new decision; do not repeat prior reasoning.\n"
            "- With a finite step budget, avoid long reasoning-only loops and transition to tool_call/response/finish as confidence grows.\n"
            "- Infer user intent from input messages before choosing next_state.\n"
            "\n"
            "Developer instructions:\n"
            f"{self._instructions}\n"
            "\n"
            f"Step: {step}\n"
            f"Input messages JSON: {self._json_dumps(self._serialize_messages(messages))}\n"
            f"Available tools JSON schema: {self._json_dumps(tool_schemas)}\n"
            f"Execution history JSON: {self._json_dumps(history)}\n"
        )

    @classmethod
    def _json_dumps(cls, value: Any) -> str:
        return json.dumps(value, ensure_ascii=True, default=cls._json_default)

    @staticmethod
    def _json_default(value: Any) -> Any:
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json")
        if isinstance(value, (datetime, date, time)):
            return value.isoformat()
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        if isinstance(value, set):
            return sorted(value, key=str)
        return str(value)

    def _build_model_messages(
        self,
        *,
        prompt: str,
        messages: list[TextMessage | ImageMessage],
    ) -> list[dict[str, Any]]:
        return [{"role": "system", "content": prompt}, *self._serialize_messages(messages)]

    def _serialize_messages(self, messages: list[TextMessage | ImageMessage]) -> list[dict[str, Any]]:
        serialized: list[dict[str, Any]] = []
        for message in messages:
            if not isinstance(message, (TextMessage, ImageMessage)):
                raise TypeError("messages must contain TextMessage/ImageMessage objects")
            serialized.append(to_oauth_message(message))
        return serialized

    def _normalize_schema(self, schema: Any) -> dict[str, Any]:
        normalized = self._transform_schema_subset(self._inline_local_refs(schema))
        self._assert_schema_root_object(normalized)
        self._assert_no_unsupported_keywords(normalized)
        self._assert_object_consistency(normalized)
        if not isinstance(normalized, dict):
            raise LLMOutputError("normalized schema must be a dict")
        return normalized

    def _inline_local_refs(self, schema: Any) -> Any:
        root = copy.deepcopy(schema)

        def resolve_pointer(pointer: str) -> Any:
            if not pointer.startswith("#/"):
                raise LLMOutputError(f"unsupported $ref pointer: {pointer}")

            node: Any = root
            for raw_part in pointer[2:].split("/"):
                part = raw_part.replace("~1", "/").replace("~0", "~")
                if not isinstance(node, dict) or part not in node:
                    raise LLMOutputError(f"unresolvable $ref pointer: {pointer}")
                node = node[part]
            return copy.deepcopy(node)

        def walk(node: Any, ref_stack: tuple[str, ...] = ()) -> Any:
            if isinstance(node, dict):
                ref = node.get("$ref")
                if isinstance(ref, str):
                    if ref in ref_stack:
                        chain = " -> ".join((*ref_stack, ref))
                        raise LLMOutputError(f"circular $ref detected: {chain}")

                    target = resolve_pointer(ref)
                    if not isinstance(target, dict):
                        raise LLMOutputError(f"$ref pointer does not resolve to object: {ref}")

                    merged = target
                    for key, value in node.items():
                        if key == "$ref":
                            continue
                        merged[key] = walk(value, ref_stack)
                    return walk(merged, (*ref_stack, ref))

                return {key: walk(value, ref_stack) for key, value in node.items()}

            if isinstance(node, list):
                return [walk(item, ref_stack) for item in node]

            return node

        return walk(root)

    def _transform_schema_subset(self, node: Any) -> Any:
        if isinstance(node, dict):
            transformed: dict[str, Any] = {}
            for key, value in node.items():
                if key in {"$defs", "definitions", "const"}:
                    continue

                out_key = "anyOf" if key == "oneOf" else key
                transformed[out_key] = self._transform_schema_subset(value)

            if "const" in node and "enum" not in transformed:
                transformed["enum"] = [node["const"]]

            if transformed.get("type") == "object":
                properties = transformed.get("properties")
                if not isinstance(properties, dict):
                    properties = {}

                transformed["properties"] = properties
                transformed["required"] = list(properties.keys())
                transformed["additionalProperties"] = False

            return transformed

        if isinstance(node, list):
            return [self._transform_schema_subset(item) for item in node]

        return node

    def _assert_no_unsupported_keywords(self, node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in _UNSUPPORTED_SCHEMA_KEYWORDS:
                    raise LLMOutputError(f"schema contains unsupported keyword: {key}")
                self._assert_no_unsupported_keywords(value)
            return

        if isinstance(node, list):
            for item in node:
                self._assert_no_unsupported_keywords(item)

    def _assert_object_consistency(self, node: Any, path: str = "$") -> None:
        if isinstance(node, dict):
            if node.get("type") == "object":
                properties = node.get("properties")
                required = node.get("required")
                additional_properties = node.get("additionalProperties")

                if not isinstance(properties, dict):
                    raise LLMOutputError(f"object schema missing properties at {path}")
                if not isinstance(required, list):
                    raise LLMOutputError(f"object schema missing required list at {path}")
                if required != list(properties.keys()):
                    raise LLMOutputError(
                        f"object schema required/properties mismatch at {path}: {required} vs {list(properties.keys())}"
                    )
                if additional_properties is not False:
                    raise LLMOutputError(
                        f"object schema must set additionalProperties=false at {path}"
                    )

            for key, value in node.items():
                self._assert_object_consistency(value, f"{path}.{key}")
            return

        if isinstance(node, list):
            for idx, item in enumerate(node):
                self._assert_object_consistency(item, f"{path}[{idx}]")

    @staticmethod
    def _assert_schema_root_object(schema: Any) -> None:
        if not isinstance(schema, dict):
            raise LLMOutputError("schema root must be an object")
