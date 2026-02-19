# Fabrix API 사용 가이드

Language: [English](api.md) | 한국어  
Home: [README](../README.md) | [README.ko.md](../README.ko.md)

## 목적 및 범위

이 문서는 Fabrix의 public 사용 표면을 설명합니다.
에이전트 생성, `run_stream`, 메시지 모델, 이벤트 스트림 처리를 다룹니다.

## 요구 사항

- Python `>=3.12`
- 패키지 설치: `pip install fabrix-ai`
- `oauth-codex` 인증 설정
- Async 런타임 (`asyncio`)

## 공개 Import

```python
from fabrix import Agent
from fabrix.events import (
    AgentEvent,
    ReasoningEvent,
    ResponseEvent,
    TaskFailedEvent,
    ToolEvent,
)
from fabrix.messages import ImageMessage, TextMessage
from fabrix.tools import ToolImagePart, ToolJSONPart, ToolOutput, ToolTextPart
```

## Agent 생성

```python
Agent(
    *,
    instructions: str,
    model: str = "gpt-5.3-codex",
    tools: list[Callable[..., Any]] | None = None,
)
```

참고:

- 내부 실행 기본값: `max_steps=128`
- public 생성자에 per-tool timeout 없음
- 호환되지 않는 tool schema는 생성 시 즉시 실패

## 메시지 모델

Fabrix 런타임 입력은 메시지 모델 객체 리스트입니다.

```python
class TextMessage(BaseModel):
    role: str = "user"
    text: str

class ImageMessage(BaseModel):
    role: str = "user"
    image: str | Path | bytes
    text: str | None = None
```

규칙:

- `run_stream`은 `TextMessage` / `ImageMessage` 인스턴스만 허용
- 생성 시 정의되지 않은 필드는 거부됩니다.
- `ImageMessage.image`는 URL/path/bytes 지원
- `bytes`와 로컬 path는 모델 호출 전에 data URL로 변환

## 스트림 실행

```python
run_stream(
    *,
    messages: list[TextMessage | ImageMessage],
) -> AsyncIterator[AgentEvent]
```

검증:

- 빈 리스트 -> `ValueError`
- 메시지 모델 외 객체 포함 -> `TypeError`

텍스트 예시:

```python
messages = [TextMessage(text="Use add_numbers to compute 3 + 9")]
async for event in agent.run_stream(messages=messages):
    ...
```

멀티모달 예시:

```python
messages = [
    TextMessage(text="이 이미지를 설명해줘"),
    ImageMessage(image="https://example.com/cat.png"),
    TextMessage(text="경고 문구를 중심으로 요약해줘"),
]
async for event in agent.run_stream(messages=messages):
    ...
```

## Tool 정의 규칙

허용되는 tool 형태:

```python
def tool(payload: BaseModel) -> ToolOutput: ...
```

런타임 규칙:

- 파라미터는 정확히 1개
- 파라미터 타입은 Pydantic `BaseModel`
- 반환 타입은 반드시 `ToolOutput`
- 인자는 payload 필드와 일치하는 JSON object
- 추가 키는 거부

`ToolOutput`은 멀티모달 part를 지원합니다.

- `ToolTextPart(type="text", text="...")`
- `ToolImagePart(type="image", image_url="...", caption=None)`
- `ToolJSONPart(type="json", data={...})`
- `ToolImagePart.image_url`은 URL/path/bytes 입력을 허용합니다.
- `http(s)`/`data:` 값은 그대로 유지됩니다.
- `file://`, 로컬 path, bytes는 로컬 절대경로 참조로 정규화됩니다.
- tool-call 인자 strict 검증은 `output_schema` + `strict_output=True`로 보장됩니다.
- 정책 프롬프트와 런타임 컨텍스트를 중복 주입하지 않으며, 런타임 제어 정보는 마지막 control message로 전달됩니다.
- 히스토리를 모델 입력으로 직렬화할 때 reasoning/tool_call/response(레거시 `tool_result` 포함) 기록을 유지하고, 로컬 이미지 참조는 다시 data URL로 변환됩니다.

## 이벤트 레퍼런스

공통 필드:

- `event_type: str`
- `step: int`
- `timestamp: datetime` (UTC)

이벤트별 핵심 필드:

| event_type | Model | Key fields |
| --- | --- | --- |
| `reasoning` | `ReasoningEvent` | `reasoning`, `focus`, `next_state` |
| `tool` | `ToolEvent` | `phase`, `tool_name`, `call_id`, `arguments`, `result` |
| `response` | `ResponseEvent` | `response`, `parts` |
| `task_failed` | `TaskFailedEvent` | `error_code`, `message` |

`ResponseEvent.response`와 `ResponseEvent.parts`는 모두 optional이며, 둘 다 `None`일 수 있습니다.

종료 규칙:

- 정상 완료는 `response` 상태에서 `next_state=null`로 `response` 이벤트를 내보낸 직후 종료됩니다.

## 실패 처리

종료 실패는 `TaskFailedEvent`로 전달됩니다.
현재 error code 예시:

- `llm_error`
- `invalid_state_type`

tool 실패는 non-terminal이며 `ToolEvent(phase="finish")`의 `result.ok == False`로 전달됩니다.
성공한 경우 `result.output`은 `ToolOutput` 타입으로 제공됩니다.

## 마이그레이션 (브레이킹)

제거된 API:

- `run_task_stream(task, images, context)`

신규 API:

- `run_stream(messages=[...])`

입력 매핑:

- `task` -> `TextMessage(text="...")`
- `images` -> `ImageMessage(image="..." | Path(...) | b"...")`
- `context` -> 직렬화해서 `TextMessage.text`에 포함

tool 반환값 매핑:

- 이전: scalar/string/dict-like 값을 직접 반환
- 현재: `ToolOutput.text(...)`, `ToolOutput.json(...)`, `ToolOutput.image(...)` 형태로 반환

## 트러블슈팅

- `ValueError: messages must be a non-empty list of TextMessage/ImageMessage objects`:
  메시지 모델 객체를 최소 1개 이상 전달하세요.
- `TypeError: messages must be a list of TextMessage/ImageMessage objects`:
  dict 대신 메시지 모델 인스턴스를 전달하세요.
- terminal 이벤트 없이 스트림이 종료되는 경우:
  마지막 `response` 상태의 `next_state`를 `null`로 설정했는지 확인하세요.
- `task_failed` + `llm_error`:
  인증/모델 설정과 tool schema 호환성을 확인하세요.
