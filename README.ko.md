# Fabrix

Language: [English](README.md) | 한국어  
API Guides: [English](docs/api.md) | [한국어](docs/api.ko.md)

## 개요

Fabrix는 `oauth-codex>=2.2.0` 위에서 동작하는 그래프 기반 에이전트 프레임워크입니다.
도구 중심 워크플로를 위해 구조화된 실행 그래프와 스트리밍 이벤트를 제공합니다.

## 핵심 기능

- 그래프 기반 4-상태 실행: `reasoning`, `tool_call`, `response`, `finish`
- Pydantic 모델 기반의 구조화된 상태 출력
- 엄격한 페이로드 검증을 포함한 순차적 도구 실행
- 단계별 관측이 가능한 async 스트리밍 이벤트 API

## 설치

```bash
pip install fabrix-ai
```

## 빠른 시작

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

## Tool 계약

Fabrix는 아래 형태의 도구를 허용합니다.

```python
def tool(payload: BaseModel) -> Any: ...
```

- 도구는 파라미터를 정확히 1개만 받아야 합니다.
- 파라미터 타입은 Pydantic `BaseModel`이어야 합니다.
- 런타임 인자는 payload 필드와 일치하는 JSON object여야 합니다.
- 추가 인자 키는 허용되지 않습니다.
- sync/async 도구를 모두 지원합니다.

## 이벤트 스트림

`run_task_stream(...)`은 다음 이벤트 타입을 생성합니다.

- `reasoning`
- `tool` (`phase="start"` / `phase="finish"`)
- `response`
- `task_finished`
- `task_failed`

일반적인 흐름:

1. `reasoning`
2. 0개 이상의 `tool` start/finish 쌍
3. 선택적 `response`
4. `task_finished` 또는 `task_failed`

## 문서

- API usage guide (English): [`docs/api.md`](docs/api.md)
- API 사용 가이드 (한국어): [`docs/api.ko.md`](docs/api.ko.md)
- English README: [`README.md`](README.md)

## 예제

- Minimal quickstart: [`examples/minimal/quickstart.py`](examples/minimal/quickstart.py)
- Data workflow: [`examples/advanced/data_workflow.py`](examples/advanced/data_workflow.py)
- Incident response workflow: [`examples/advanced/incident_response.py`](examples/advanced/incident_response.py)

## 참고 사항

- 공개 런타임 진입점은 `fabrix.Agent`입니다.
- 실행 기본값은 내부 고정값입니다: `max_steps=24`, public per-tool timeout 옵션 없음.
- `max_steps`에 도달했을 때 최소 1회 `response`가 있었으면 마지막 응답으로 `task_finished`를 내보내며 `completion_reason="max_steps_reached"`를 설정합니다.
- 응답/최종 출력 없이 `max_steps`에 도달하면 `task_failed`와 `error_code="max_steps_reached"`로 종료됩니다.
