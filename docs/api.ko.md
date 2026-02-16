# Fabrix API 사용 가이드

Language: [English](api.md) | 한국어  
Home: [README](../README.md) | [README.ko.md](../README.ko.md)

## 목적 및 범위

이 문서는 Fabrix의 public 사용 표면을 설명합니다.
`Agent` 생성, `run_task_stream` 실행, tool 정의, 스트리밍 이벤트 처리 방법을 다룹니다.
내부 구현 상세는 관측 가능한 동작에 영향을 주는 경우를 제외하고 범위에서 제외합니다.

## 요구 사항

- Python `>=3.12`
- 패키지 설치: `pip install fabrix-ai`
- 실제 실행 시 모델 호출을 위한 `oauth-codex` 인증 설정
- Async 런타임 (예제는 `asyncio` 사용)

## 공개 Import

```python
from fabrix import Agent
from fabrix.events import (
    AgentEvent,
    ReasoningEvent,
    ResponseEvent,
    TaskFailedEvent,
    TaskFinishedEvent,
    ToolEvent,
)
```

Public 런타임 진입점:

- `fabrix.Agent`

스트림 소비자를 위한 public 이벤트 모델:

- `fabrix.events.AgentEvent`
- `fabrix.events.ReasoningEvent`
- `fabrix.events.ToolEvent`
- `fabrix.events.ResponseEvent`
- `fabrix.events.TaskFinishedEvent`
- `fabrix.events.TaskFailedEvent`

## Agent 생성

생성자 시그니처:

```python
Agent(
    *,
    instructions: str,
    model: str = "gpt-5.3-codex",
    tools: list[Callable[..., Any]] | None = None,
)
```

파라미터 동작:

- `instructions`: 필수 개발자 지시문 텍스트.
- `model`: provider에 전달되는 모델 이름. 기본값은 `"gpt-5.3-codex"`.
- `tools`: 선택적 tool callable 목록이며, 생성 시점에 검증됩니다.

참고:

- 실행 기본값은 내부 고정값이며 `max_steps=24`입니다.
- public 생성자에 per-tool timeout 파라미터는 없습니다.
- 호환되지 않는 tool schema는 에이전트 생성 시 즉시 실패합니다.

## Tool 정의 규칙

허용되는 tool 형태:

```python
def tool(payload: BaseModel) -> Any: ...
```

런타임이 강제하는 규칙:

- 파라미터는 정확히 1개여야 합니다.
- 파라미터 타입은 Pydantic `BaseModel` 하위 클래스여야 합니다.
- positional-only 파라미터는 허용되지 않습니다.
- 가변 인자 `*args`, `**kwargs`는 허용되지 않습니다.
- 파라미터 기본값은 허용되지 않습니다.
- sync/async callable 모두 지원됩니다.

런타임 인자 검증:

- tool 인자는 JSON object여야 합니다.
- 인자 키는 payload 모델 필드와 일치해야 합니다.
- 추가 키가 있으면 `"unexpected tool arguments: extra"` 같은 오류를 반환합니다.

## 작업 실행

메서드 시그니처:

```python
run_task_stream(
    task: str,
    *,
    context: dict[str, Any] | None = None,
) -> AsyncIterator[AgentEvent]
```

사용 시 참고:

- `task`는 사용자 작업 문자열입니다.
- `context`는 각 단계에서 모델에 전달되는 선택적 구조화 데이터입니다.
- 스트림은 종료 이벤트가 발생할 때까지 이벤트를 순차적으로 반환합니다.

예시:

```python
async for event in agent.run_task_stream(
    "Analyze context.raw_rows and return a summary",
    context={"raw_rows": [{"category": "A", "value": 3.2}]},
):
    ...
```

## 스트리밍 이벤트 처리

권장 분기 패턴:

```python
async for event in agent.run_task_stream("Use add_numbers to compute 3 + 9"):
    if isinstance(event, ReasoningEvent):
        print(event.reasoning, event.focus, event.next_state)
    elif isinstance(event, ToolEvent):
        if event.phase == "start":
            print("calling", event.tool_name, event.arguments)
        elif event.result is not None:
            print("tool ok:", event.result.ok, "error:", event.result.error)
    elif isinstance(event, ResponseEvent):
        print(event.response)
    elif isinstance(event, TaskFinishedEvent):
        print(event.final_output, event.completion_reason)
    elif isinstance(event, TaskFailedEvent):
        print(event.error_code, event.message)
```

## 이벤트 레퍼런스

모든 이벤트 공통 필드:

- `event_type: str`
- `step: int` (현재 executor는 1부터 시작)
- `timestamp: datetime` (UTC)

이벤트별 핵심 필드:

| event_type | Model | Key fields |
| --- | --- | --- |
| `reasoning` | `ReasoningEvent` | `reasoning`, `focus`, `next_state` |
| `tool` | `ToolEvent` | `phase`, `tool_name`, `call_id`, `arguments`, `result` |
| `response` | `ResponseEvent` | `response` |
| `task_finished` | `TaskFinishedEvent` | `final_output`, `completion_reason` |
| `task_failed` | `TaskFailedEvent` | `error_code`, `message` |

`ToolEvent.result`는 `phase="finish"`에서 채워지며 다음 필드를 포함합니다.

- `ok: bool`
- `output: Any | None`
- `error: str | None`
- `latency_ms: float`

## 실패 처리

종료 실패는 `TaskFailedEvent`로 전달됩니다.
현재 `error_code` 값은 다음을 포함합니다.

- `llm_error`: 모델/provider가 유효한 구조화 상태를 생성하지 못함
- `invalid_state_type`: 모델이 현재 기대 상태와 다른 state type을 반환함
- `invalid_transition`: 상태 전이가 그래프 규칙을 위반함
- `max_steps_reached`: 24단계 내에 response/final output이 생성되지 않음

tool 호출 실패는 단독으로 종료 조건이 아니며 `ToolEvent(phase="finish")`에서 `result.ok == False`로 전달됩니다.
일반적인 tool 오류:

- `tool not found: <name>`
- `tool arguments must be a JSON object`
- `unexpected tool arguments: <extra_key>`

## 동작 보장

- 각 실행은 `reasoning` 상태에서 시작합니다.
- 하나의 `tool_call` 상태 안에서 tool 호출은 명시된 순서대로 순차 실행됩니다.
- 유효하지 않은 상태 전이는 `task_failed`로 즉시 종료됩니다.
- `finish` 상태가 생성되면 `task_finished`를 내보내고 스트림이 종료됩니다.
- step 제한에 도달했을 때 최소 1회 `response`가 있으면 마지막 response를 사용해 `task_finished`를 내보내고 `completion_reason="max_steps_reached"`를 설정합니다.
- response/final output 없이 step 제한에 도달하면 `task_failed`와 `error_code="max_steps_reached"`로 종료합니다.

## 전체 예제

```python
import asyncio

from pydantic import BaseModel

from fabrix import Agent
from fabrix.events import (
    ReasoningEvent,
    ResponseEvent,
    TaskFailedEvent,
    TaskFinishedEvent,
    ToolEvent,
)


class AddInput(BaseModel):
    a: int
    b: int


def add_numbers(payload: AddInput) -> int:
    return payload.a + payload.b


async def main() -> None:
    agent = Agent(
        instructions="You are a precise assistant.",
        model="gpt-5.3-codex",
        tools=[add_numbers],
    )

    async for event in agent.run_task_stream("Use add_numbers to compute 3 + 9"):
        print(f"[step={event.step}] {event.event_type}")

        if isinstance(event, ReasoningEvent):
            print("reasoning:", event.reasoning)
            print("focus:", event.focus)
        elif isinstance(event, ToolEvent):
            if event.phase == "start":
                print("tool call:", event.tool_name, event.arguments)
            elif event.result is not None:
                print("tool result:", event.result.model_dump())
        elif isinstance(event, ResponseEvent):
            print("response:", event.response)
        elif isinstance(event, TaskFinishedEvent):
            print("completion reason:", event.completion_reason)
            print("final:", event.final_output)
        elif isinstance(event, TaskFailedEvent):
            print("failed:", event.error_code, event.message)


asyncio.run(main())
```

## 트러블슈팅

- tool 시그니처 관련 `TypeError`:
  도구를 Pydantic payload 파라미터 1개 형태로 정의하세요.
- `unexpected tool arguments: ...`:
  모델이 생성한 인자 키가 payload 모델 필드명과 정확히 일치하는지 확인하세요.
- `tool arguments must be a JSON object`:
  tool 인자가 object 형태 JSON인지 확인하세요.
- `task_failed` + `invalid_transition`:
  모델이 허용 전이를 따르도록 지시문을 더 구체화하세요.
- `task_failed` + `llm_error`:
  모델/인증 설정과 tool schema의 strict JSON schema 변환 호환성을 점검하세요.
