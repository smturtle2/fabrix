from __future__ import annotations

import copy
import json
from collections.abc import Callable
from datetime import date, datetime, time
from enum import Enum
from typing import Any

from oauth_codex import OAuthCodexClient
from oauth_codex.tooling import build_strict_response_format
from pydantic import BaseModel, ValidationError

from fabrix.errors import LLMOutputError, RetryableLLMOutputError
from fabrix.graph.state import (
    MAX_TOOL_CALLS_PER_STEP,
    NextState,
    ReasoningState,
    ResponseState,
    StateEnvelope,
    ToolCallState,
)
from fabrix.graph.transitions import allowed_next_states
from fabrix.media import coerce_image_to_url
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
    type[ReasoningState | ToolCallState | ResponseState],
] = {
    NextState.reasoning: ReasoningState,
    NextState.tool_call: ToolCallState,
    NextState.response: ResponseState,
}


class OAuthCodexStateProvider:
    def __init__(
        self,
        *,
        instructions: str | Callable[[], str],
        client: OAuthCodexClient | None = None,
        model: str = DEFAULT_MODEL,
        reasoning_effort: ReasoningEffort = "low",
    ) -> None:
        self._client = client or OAuthCodexClient()
        self._instructions = instructions
        self._model = model
        self._reasoning_effort = reasoning_effort
        self._output_schema_cache: dict[tuple[str, str], dict[str, Any]] = {}

    def validate_tool_schemas(self, tool_schemas: list[dict[str, Any]]) -> None:
        if tool_schemas:
            self._build_output_schema(
                current_state=NextState.tool_call,
                tool_schemas=tool_schemas,
            )

        for state in (NextState.reasoning, NextState.response):
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
        serialized_messages = self._serialize_messages(messages)
        serialized_history_messages = self._serialize_history_messages(history)
        developer_instructions = self._resolve_instructions()
        prompt = self._build_prompt(has_tools=bool(tool_schemas))
        control_message = self._build_control_message(
            current_state=current_state,
            step=step,
            tool_schemas=tool_schemas,
            developer_instructions=developer_instructions,
        )
        output_schema = self._build_output_schema(
            current_state=current_state,
            tool_schemas=tool_schemas,
        )

        try:
            payload = await self._client.agenerate(
                messages=self._build_model_messages(
                    prompt=prompt,
                    input_messages=serialized_messages,
                    history_messages=serialized_history_messages,
                    control_message=control_message,
                ),
                model=self._model,
                reasoning_effort=self._reasoning_effort,
                output_schema=output_schema,
                strict_output=True,
            )
        except Exception as exc:
            raise RetryableLLMOutputError(
                f"failed to produce valid state output: {exc}"
            ) from exc

        return self._parse_payload(payload)

    def _parse_payload(self, payload: str | dict[str, Any]) -> StateEnvelope:
        if isinstance(payload, str):
            try:
                data = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise RetryableLLMOutputError(
                    "model returned non-JSON text for structured state"
                ) from exc
        else:
            data = payload
        try:
            return StateEnvelope.model_validate(data)
        except ValidationError as exc:
            raise RetryableLLMOutputError(f"state envelope validation failed: {exc}") from exc

    def _build_output_schema(
        self,
        *,
        current_state: NextState,
        tool_schemas: list[dict[str, Any]],
    ) -> dict[str, Any]:
        cache_key = (
            current_state.value,
            self._tool_schema_fingerprint(tool_schemas),
        )
        cached = self._output_schema_cache.get(cache_key)
        if cached is not None:
            return copy.deepcopy(cached)

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
        normalized_output_schema = self._normalize_schema(output_schema)
        self._output_schema_cache[cache_key] = copy.deepcopy(normalized_output_schema)
        return copy.deepcopy(normalized_output_schema)

    def _tool_schema_fingerprint(self, tool_schemas: list[dict[str, Any]]) -> str:
        return self._json_dumps(tool_schemas, sort_keys=True)

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
        self._apply_next_state_values(
            schema=next_state_schema,
            values=[state.value for state in allowed_next_states(current_state)],
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

    def _apply_next_state_values(self, *, schema: dict[str, Any], values: list[str]) -> None:
        if "enum" in schema:
            schema["enum"] = values
            return

        any_of = schema.get("anyOf")
        if isinstance(any_of, list):
            for candidate in any_of:
                if not isinstance(candidate, dict):
                    continue
                if candidate.get("type") == "null":
                    continue
                candidate["enum"] = values
                return

        raise LLMOutputError("state schema next_state must define enum or anyOf")

    def _render_transition_rules(self) -> str:
        grouped_rules: dict[tuple[str, ...], list[str]] = {}
        for state in NextState:
            allowed = [next_state.value for next_state in allowed_next_states(state)]
            if state is NextState.response:
                allowed.append("null")
            grouped_rules.setdefault(tuple(allowed), []).append(state.value)

        rules: list[str] = []
        for allowed, from_states in grouped_rules.items():
            rules.append(
                f"{{ {' | '.join(from_states)} }} -> {{ {' | '.join(allowed)} }}"
            )

        return "; ".join(rules)

    def _build_prompt(self, *, has_tools: bool) -> str:
        no_tools_line = (
            "No tools are registered. Avoid choosing `next_state=tool_call`.\n"
            if not has_tools
            else ""
        )

        return (
            "You are a graph-based agent state generator.\n"
            "Return ONLY a JSON object matching the provided schema.\n"
            f"Transition rules: {self._render_transition_rules()}.\n"
            "Decision policy:\n"
            "- Infer user intent, constraints, and missing information from input messages and history before selecting `next_state`.\n"
            "- Runtime control context is provided in the final user message prefixed with `state_control:`.\n"
            "- The `state_control` payload is authoritative for `current_state`, `step`, allowed transitions, tools, and developer instructions.\n"
            "Tool usage rules:\n"
            "- You MUST choose `tool_call` state only when external computation/data access is required.\n"
            "- In `tool_call` state, each `arguments` object must exactly match the selected tool schema.\n"
            f"- In `tool_call` state, include between 1 and {MAX_TOOL_CALLS_PER_STEP} `tool_calls`.\n"
            f"{no_tools_line}"
            "Response rules:\n"
            "- `response` state may emit plain text (`response`), structured parts, both, or neither.\n"
            "- For image output, prefer using `parts` with `type=image`.\n"
            "- Empty responses are allowed (`response=null` and `parts=null`).\n"
            "- In `response` state, you MUST set `next_state=null` when the task is complete.\n"
            "Reasoning loop strategy:\n"
            "- Treat `reasoning` as a short decision journal, not free-form commentary.\n"
            "- Use `reasoning` to record what changed in this step (new evidence or decision delta), and use `focus` to define the immediate next check/action.\n"
            "- `focus` and `next_state` MUST be aligned: with `next_state=reasoning`, `focus` should be the next question/validation target; when transitioning, `focus` should be the immediate execution objective.\n"
            "- Prefer English in `reasoning` and `focus` for consistency; keep each reasoning step to 1-2 sentences.\n"
            "- Consider a task non-trivial when it involves multiple tools, competing constraints/objectives, ranking, or scenario comparison.\n"
            "- For non-trivial tasks, do not transition on the same reasoning step where you first form a candidate decision; use the next reasoning step as a brief challenge pass.\n"
            "- End reasoning only when you can state a concrete transition reason in `reasoning` (enough evidence, clear next executable action, and acceptable residual uncertainty).\n"
            "- If that challenge pass invalidates the transition reason, continue reasoning and refine the decision basis; if not, you may transition.\n"
            "- Transition to `tool_call`/`response` only after that reason is explicit.\n"
        )

    def _resolve_instructions(self) -> str:
        resolved = self._instructions() if callable(self._instructions) else self._instructions
        if not isinstance(resolved, str):
            raise TypeError("instructions must be a string or a callable returning a string")
        return resolved

    @classmethod
    def _json_dumps(cls, value: Any, *, sort_keys: bool = False) -> str:
        return json.dumps(
            value,
            ensure_ascii=False,
            default=cls._json_default,
            separators=(",", ":"),
            sort_keys=sort_keys,
        )

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
        input_messages: list[dict[str, Any]],
        history_messages: list[dict[str, Any]],
        control_message: dict[str, Any],
    ) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": prompt},
            *input_messages,
            *history_messages,
            control_message,
        ]

    def _build_control_message(
        self,
        *,
        current_state: NextState,
        step: int,
        tool_schemas: list[dict[str, Any]],
        developer_instructions: str,
    ) -> dict[str, Any]:
        tool_names = [
            name
            for schema in tool_schemas
            if isinstance(schema, dict)
            for name in [schema.get("name")]
            if isinstance(name, str) and name
        ]
        payload = {
            "current_state": current_state.value,
            "step": step,
            "allowed_next_states": [
                *[next_state.value for next_state in allowed_next_states(current_state)],
                *(["null"] if current_state is NextState.response else []),
            ],
            "tool_names": tool_names,
            "developer_instructions": developer_instructions,
        }
        return {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": f"state_control:{self._json_dumps(payload)}",
                }
            ],
        }

    def _serialize_messages(self, messages: list[TextMessage | ImageMessage]) -> list[dict[str, Any]]:
        serialized: list[dict[str, Any]] = []
        for message in messages:
            if not isinstance(message, (TextMessage, ImageMessage)):
                raise TypeError("messages must contain TextMessage/ImageMessage objects")
            serialized.append(to_oauth_message(message))
        return serialized

    def _serialize_history_messages(self, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        serialized: list[dict[str, Any]] = []
        for index, item in enumerate(history):
            if not isinstance(item, dict):
                serialized.append(self._serialize_unknown_history_item(item, index=index))
                continue

            kind = item.get("kind")
            if kind == "reasoning":
                serialized.append(self._serialize_reasoning_history(item))
                continue

            if kind == "tool_result":
                message = self._serialize_tool_result_history(item)
                if message is not None:
                    serialized.append(message)
                continue

            if kind == "tool_call":
                serialized.extend(self._serialize_tool_call_history(item))
                continue

            if kind == "response":
                serialized.append(self._serialize_response_history(item))
                continue

            serialized.append(self._serialize_unknown_history_item(item, index=index))
        return serialized

    def _serialize_reasoning_history(self, item: dict[str, Any]) -> dict[str, Any]:
        step = item.get("step")
        focus = item.get("focus")
        next_state = item.get("next_state")
        reasoning = item.get("reasoning")
        summary = (
            f"reasoning step={step} focus={focus or 'unknown'} "
            f"next_state={next_state or 'unknown'}"
        )
        text = reasoning if isinstance(reasoning, str) and reasoning.strip() else ""
        return {
            "role": "assistant",
            "content": f"{summary}\n{text}".strip(),
        }

    def _serialize_tool_call_history(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        step = item.get("step")
        next_state = item.get("next_state")
        tool_calls = item.get("tool_calls")
        tool_results = item.get("tool_results")

        call_count = len(tool_calls) if isinstance(tool_calls, list) else 0
        result_count = len(tool_results) if isinstance(tool_results, list) else 0
        parts: list[dict[str, str]] = [
            {
                "type": "input_text",
                "text": (
                    f"tool_call step={step} next_state={next_state or 'unknown'} "
                    f"calls={call_count} results={result_count}"
                ),
            }
        ]

        if isinstance(tool_calls, list):
            for call in tool_calls:
                if not isinstance(call, dict):
                    parts.append(
                        {
                            "type": "input_text",
                            "text": f"planned_tool_call raw={self._json_dumps(call)}",
                        }
                    )
                    continue
                name = call.get("name")
                call_id = call.get("call_id")
                why = call.get("why")
                arguments = call.get("arguments")
                text = (
                    f"planned_tool_call name={name or 'unknown'} "
                    f"call_id={call_id or 'unknown'} arguments={self._json_dumps(arguments)}"
                )
                if isinstance(why, str) and why.strip():
                    text = f"{text} why={why.strip()}"
                parts.append({"type": "input_text", "text": text})

        serialized: list[dict[str, Any]] = [{"role": "user", "content": parts}]
        if isinstance(tool_results, list):
            for result in tool_results:
                if not isinstance(result, dict):
                    serialized.append(
                        self._history_text_message(
                            f"tool_result raw={self._json_dumps(result)}"
                        )
                    )
                    continue
                message = self._serialize_tool_result_history(result)
                if message is not None:
                    serialized.append(message)
        return serialized

    def _serialize_response_history(self, item: dict[str, Any]) -> dict[str, Any]:
        step = item.get("step")
        next_state = item.get("next_state")
        response = item.get("response")
        raw_parts = item.get("parts")

        content_parts: list[dict[str, str]] = []
        if isinstance(response, str) and response:
            content_parts.append({"type": "output_text", "text": response})

        if isinstance(raw_parts, list):
            for raw_part in raw_parts:
                if not isinstance(raw_part, dict):
                    content_parts.append(
                        {"type": "output_text", "text": self._json_dumps(raw_part)}
                    )
                    continue

                part_type = raw_part.get("type")
                if part_type == "text":
                    text = raw_part.get("text")
                    if isinstance(text, str) and text:
                        content_parts.append({"type": "output_text", "text": text})
                    continue

                if part_type == "json":
                    content_parts.append(
                        {
                            "type": "output_text",
                            "text": self._json_dumps(raw_part.get("data")),
                        }
                    )
                    continue

                if part_type == "image":
                    caption = raw_part.get("caption")
                    if isinstance(caption, str) and caption.strip():
                        content_parts.append(
                            {"type": "output_text", "text": caption.strip()}
                        )
                    image_url = raw_part.get("image_url")
                    if isinstance(image_url, str) and image_url.strip():
                        try:
                            normalized_image_url = coerce_image_to_url(image_url.strip())
                        except (FileNotFoundError, TypeError, ValueError) as exc:
                            content_parts.append(
                                {
                                    "type": "output_text",
                                    "text": f"[response image skipped: {exc}]",
                                }
                            )
                        else:
                            content_parts.append(
                                {
                                    "type": "input_image",
                                    "image_url": normalized_image_url,
                                }
                            )
                    continue

                content_parts.append(
                    {"type": "output_text", "text": self._json_dumps(raw_part)}
                )

        if not content_parts:
            content_parts.append(
                {
                    "type": "output_text",
                    "text": (
                        f"response step={step} next_state={next_state} "
                        "response=null parts=null"
                    ),
                }
            )

        return {"role": "assistant", "content": content_parts}

    def _serialize_unknown_history_item(self, item: Any, *, index: int) -> dict[str, Any]:
        return self._history_text_message(
            f"history_item index={index} raw={self._json_dumps(item)}"
        )

    @staticmethod
    def _history_text_message(text: str) -> dict[str, Any]:
        return {"role": "user", "content": [{"type": "input_text", "text": text}]}

    def _serialize_tool_result_history(self, item: dict[str, Any]) -> dict[str, Any] | None:
        step = item.get("step")
        tool_name = item.get("tool_name")
        call_id = item.get("call_id")
        ok = item.get("ok")

        if ok is not True:
            error = item.get("error")
            summary = (
                f"tool_result step={step} tool={tool_name} call_id={call_id} "
                f"ok=False error={error or 'unknown'}"
            )
            return {"role": "user", "content": [{"type": "input_text", "text": summary}]}

        parts: list[dict[str, str]] = [
            {
                "type": "input_text",
                "text": f"tool_result step={step} tool={tool_name} call_id={call_id} ok=True",
            }
        ]
        output = item.get("output")
        if isinstance(output, dict):
            output_parts = output.get("parts")
            if isinstance(output_parts, list):
                for output_part in output_parts:
                    if not isinstance(output_part, dict):
                        continue
                    part_type = output_part.get("type")
                    if part_type == "text":
                        text = output_part.get("text")
                        if isinstance(text, str) and text:
                            parts.append({"type": "input_text", "text": text})
                        continue
                    if part_type == "json":
                        parts.append(
                            {
                                "type": "input_text",
                                "text": self._json_dumps(output_part.get("data")),
                            }
                        )
                        continue
                    if part_type == "image":
                        caption = output_part.get("caption")
                        if isinstance(caption, str) and caption.strip():
                            parts.append({"type": "input_text", "text": caption})
                        image_url = output_part.get("image_url")
                        if isinstance(image_url, str) and image_url.strip():
                            try:
                                normalized_image_url = coerce_image_to_url(image_url.strip())
                            except (FileNotFoundError, TypeError, ValueError) as exc:
                                parts.append(
                                    {
                                        "type": "input_text",
                                        "text": f"[tool_result image skipped: {exc}]",
                                    }
                                )
                            else:
                                parts.append(
                                    {
                                        "type": "input_image",
                                        "image_url": normalized_image_url,
                                    }
                                )
                return {"role": "user", "content": parts}

        if output is not None:
            parts.append({"type": "input_text", "text": self._json_dumps(output)})
        return {"role": "user", "content": parts}

    def _normalize_schema(self, schema: Any) -> dict[str, Any]:
        normalized = self._transform_schema_subset(self._inline_local_refs(schema))
        self._assert_schema_root_object(normalized)
        self._assert_no_unsupported_keywords(normalized)
        self._assert_type_shape_consistency(normalized)
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

    def _assert_type_shape_consistency(self, node: Any, path: str = "$") -> None:
        if isinstance(node, dict):
            type_values = self._collect_schema_type_values(node.get("type"), path)

            if "array" in type_values:
                items = node.get("items")
                if not isinstance(items, dict):
                    raise LLMOutputError(f"array schema missing items at {path}")

            if "object" in type_values:
                self._assert_object_shape(node, path)

            for key, value in node.items():
                self._assert_type_shape_consistency(value, f"{path}.{key}")
            return

        if isinstance(node, list):
            for idx, item in enumerate(node):
                self._assert_type_shape_consistency(item, f"{path}[{idx}]")

    def _collect_schema_type_values(self, schema_type: Any, path: str) -> set[str]:
        if isinstance(schema_type, str):
            return {schema_type}

        if isinstance(schema_type, list):
            values: set[str] = set()
            for idx, type_name in enumerate(schema_type):
                if not isinstance(type_name, str):
                    raise LLMOutputError(f"schema type list must contain only strings at {path}.type[{idx}]")
                values.add(type_name)
            return values

        return set()

    def _assert_object_shape(self, node: dict[str, Any], path: str) -> None:
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

    def _assert_object_consistency(self, node: Any, path: str = "$") -> None:
        if isinstance(node, dict):
            if node.get("type") == "object":
                self._assert_object_shape(node, path)

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
