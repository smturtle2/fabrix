from __future__ import annotations

import random
from typing import Any

import pytest

from fabrix.errors import LLMOutputError
from fabrix.llm.oauth_codex import OAuthCodexStateProvider

_FORBIDDEN = {"$ref", "$defs", "definitions", "oneOf", "const"}


def _assert_no_keywords(node: Any) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            assert key not in _FORBIDDEN
            _assert_no_keywords(value)
        return

    if isinstance(node, list):
        for item in node:
            _assert_no_keywords(item)


def _assert_object_consistency(node: Any) -> None:
    if isinstance(node, dict):
        if node.get("type") == "object":
            properties = node.get("properties")
            required = node.get("required")
            assert isinstance(properties, dict)
            assert isinstance(required, list)
            assert required == list(properties.keys())
            assert node.get("additionalProperties") is False

        for value in node.values():
            _assert_object_consistency(value)
        return

    if isinstance(node, list):
        for item in node:
            _assert_object_consistency(item)


def _random_literal(rng: random.Random) -> Any:
    pool: list[Any] = [None, True, False, rng.randint(-10, 10), f"v{rng.randint(0, 9)}"]
    return rng.choice(pool)


def _random_schema_node(rng: random.Random, *, depth: int, def_names: list[str]) -> dict[str, Any]:
    if depth <= 0:
        leaf_options = [
            {"type": "string"},
            {"type": "integer"},
            {"const": _random_literal(rng)},
            {
                "oneOf": [
                    {"type": "string"},
                    {"const": _random_literal(rng)},
                ]
            },
        ]
        return rng.choice(leaf_options)

    kind = rng.choice(["object", "array", "oneOf", "const", "primitive", "ref"])

    if kind == "object":
        prop_count = rng.randint(0, 3)
        properties: dict[str, Any] = {}
        for idx in range(prop_count):
            properties[f"p{idx}"] = _random_schema_node(rng, depth=depth - 1, def_names=def_names)
        return {
            "type": "object",
            "properties": properties,
            "required": list(reversed(list(properties.keys()))),
            "additionalProperties": True,
        }

    if kind == "array":
        return {
            "type": "array",
            "items": _random_schema_node(rng, depth=depth - 1, def_names=def_names),
        }

    if kind == "oneOf":
        return {
            "oneOf": [
                _random_schema_node(rng, depth=depth - 1, def_names=def_names),
                _random_schema_node(rng, depth=depth - 1, def_names=def_names),
            ]
        }

    if kind == "const":
        return {"const": _random_literal(rng)}

    if kind == "ref" and def_names:
        return {"$ref": f"#/$defs/{rng.choice(def_names)}"}

    return rng.choice([{"type": "string"}, {"type": "number"}, {"type": "boolean"}])


def _build_random_schema(seed: int) -> dict[str, Any]:
    rng = random.Random(seed)

    def_names = [f"Def{i}" for i in range(rng.randint(1, 4))]
    defs: dict[str, Any] = {}
    for name in def_names:
        def_schema = _random_schema_node(rng, depth=2, def_names=[])
        if def_schema.get("type") != "object":
            def_schema = {
                "type": "object",
                "properties": {"value": def_schema},
                "required": [],
                "additionalProperties": True,
            }
        defs[name] = def_schema

    prop_count = rng.randint(1, 4)
    properties: dict[str, Any] = {}
    for idx in range(prop_count):
        properties[f"field{idx}"] = _random_schema_node(rng, depth=3, def_names=def_names)

    return {
        "type": "object",
        "properties": properties,
        "required": [],
        "additionalProperties": True,
        "$defs": defs,
    }


def test_normalization_fuzz_outputs_provider_safe_schema(fake_client: Any) -> None:
    provider = OAuthCodexStateProvider(instructions="x", client=fake_client)

    for seed in range(50):
        schema = _build_random_schema(seed)
        normalized = provider._normalize_schema(schema)

        _assert_no_keywords(normalized)
        _assert_object_consistency(normalized)


def test_normalization_rejects_unresolvable_ref(fake_client: Any) -> None:
    provider = OAuthCodexStateProvider(instructions="x", client=fake_client)

    schema = {
        "type": "object",
        "properties": {"x": {"$ref": "#/$defs/Missing"}},
        "required": ["x"],
        "additionalProperties": False,
    }

    with pytest.raises(LLMOutputError, match="unresolvable \\$ref pointer"):
        provider._normalize_schema(schema)


def test_normalization_rejects_non_local_ref(fake_client: Any) -> None:
    provider = OAuthCodexStateProvider(instructions="x", client=fake_client)

    schema = {
        "type": "object",
        "properties": {"x": {"$ref": "https://example.com/schema.json#/x"}},
        "required": ["x"],
        "additionalProperties": False,
    }

    with pytest.raises(LLMOutputError, match="unsupported \\$ref pointer"):
        provider._normalize_schema(schema)


def test_normalization_rejects_circular_refs(fake_client: Any) -> None:
    provider = OAuthCodexStateProvider(instructions="x", client=fake_client)

    schema = {
        "$defs": {
            "A": {"$ref": "#/$defs/B"},
            "B": {"$ref": "#/$defs/A"},
        },
        "type": "object",
        "properties": {
            "root": {"$ref": "#/$defs/A"},
        },
        "required": ["root"],
        "additionalProperties": False,
    }

    with pytest.raises(LLMOutputError, match="circular \\$ref detected"):
        provider._normalize_schema(schema)


def test_normalization_rejects_type_union_array_without_items(fake_client: Any) -> None:
    provider = OAuthCodexStateProvider(instructions="x", client=fake_client)

    schema = {
        "type": "object",
        "properties": {
            "value": {
                "type": ["string", "array"],
            }
        },
        "required": ["value"],
        "additionalProperties": False,
    }

    with pytest.raises(LLMOutputError, match="array schema missing items"):
        provider._normalize_schema(schema)
