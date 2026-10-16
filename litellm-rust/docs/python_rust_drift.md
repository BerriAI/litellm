# Python과 Rust 간 구현 차이 점검 설계

고정된 규칙에 따른 변경 목록 검사, 에이전트 분석, Rust 팀의 필수 리뷰를 함께 사용한다. 체크포인트에는 정해진 범위에서 어느 커밋까지의 Python 동작을 확인하고 반영했는지 기록한다. 현재 상태에서 Rust가 Python의 모든 기능을 구현했다고 선언할 수는 없다

이 제안은 [PR #46790](https://github.com/BerriAI/litellm/pull/46790) 이후 요청된 초기 점검 대상 정리와 검토 강제 방안을 다룬다. 해당 프로토타입은 Anthropic 베타 기능 두 개의 공통 타입과 판별 함수를 생성하지만, 판별 함수를 실제 운영 라우팅에 연결하지는 않았다. PR은 병합되지 않고 닫혔다. 공통 코드 생성은 범위가 명확한 계약에 유용하지만, 그것만으로 호출 조율, 동적 제공자 선택, 폴백, 스트리밍 수명 주기, 호스트 동작까지 다룰 수는 없다

아래 저장소 조사 결과는 main의 [`6530b329f43ef456c502ba2db7656f871d62474e`](https://github.com/BerriAI/litellm/tree/6530b329f43ef456c502ba2db7656f871d62474e) 커밋을 기준으로 한다. GitHub 규칙은 2026년 10월 16일에 확인했다. 이 문서는 설계 제안이며, 검사나 체크포인트를 배포하거나 코드 소유권 및 저장소 설정을 변경하지 않았다

## 현재 구현과 적용 범위

[적용 정책 카탈로그](../../litellm/rust_bridge/catalog.py)는 Anthropic, DeepSeek, Vertex AI의 Messages 호출에 대해 설정에 따라 Rust를 선택할 수 있게 한다. 나머지 Messages 호출과 Responses는 Python으로만 처리한다. 네이티브 구현의 지원 여부, Python에서의 Rust 적용 정책, 게이트웨이에서의 노출 여부는 별도로 구분해야 한다

`crates/inference-messages/src/common_utils.rs`의 네이티브 Messages 제공자 선택 로직은 적용 정책 카탈로그보다 더 많은 제공자를 지원한다. 네이티브 Responses의 직접 HTTP 호출은 OpenAI와 OpenRouter를 지원하지만, Python 바인딩은 `crates/python-bridge/src/routes/responses/host.rs`에서 HTTP 스트리밍을 거부한다. 게이트웨이는 코어의 Responses SSE를 사용할 수 있다. HTTP create, WebSocket 세션, Python의 get, delete, list-input-items, cancel, compact 작업은 각각 별도 범위로 관리해야 한다

[제공자 구현 규칙](../crates/llms/AGENTS.md)에는 Responses의 모델별 파라미터 재작성과 Responses-to-Chat 에뮬레이션을 비롯한 부분 이식 상태가 명시되어 있다. 잘못된 SSE 이벤트를 Python과 다르게 엄격하게 처리하는 의도적 차이도 기록되어 있다. 이런 항목은 전체 동작이 같다고 선언하는 대신, 식별 가능한 예외나 미구현 항목으로 남겨야 한다

[추론 계층 규칙](../crates/inference/AGENTS.md)에 따르면 기존 콜백 어댑터 경로의 사용량 및 비용 처리는 Python이 담당하며, 게이트웨이는 이를 별도로 담당한다. [네이티브 라우터](../crates/router/README.md)는 정확히 일치하는 배포를 조회하는 기본 구현으로, 재시도, 쿨다운, 폴백, 라우팅 전략, 다중 배포를 지원하지 않는다. 네이티브 엔드포인트가 있다는 사실만으로 라우팅이나 비용 처리까지 같다고 볼 수는 없다

## 초기 점검 대상

아래에서 `crates/`로 시작하는 경로는 `litellm-rust/` 기준이다. 각 행은 점검 대상의 큰 분류이며, 동작이 다르면 제공자, 작업, SDK 또는 게이트웨이 인터페이스, 스트리밍 모드별로 더 나눈다

| 점검 영역 | Python 소스 | Rust 구현 또는 경계 | 확인할 동작 |
| --- | --- | --- | --- |
| Messages 진입점 | `litellm/messages/dispatch.py`, `litellm/rust_bridge/messages/` | `crates/python-bridge/src/routes/messages/` | 공개 함수 시그니처, 동기 및 비동기 호출, 네이티브 경로 우회와 실행 허용 판단 |
| Messages 호출 조율 | `litellm/llms/anthropic/pass_through/messages/handler.py` | `crates/inference-messages/src/` | 제공자 선택, 준비 순서, 인증 정보, 헤더, 훅, 캐시, 수명 주기 |
| Messages 변환 | `litellm/llms/base_llm/anthropic_messages/`, `litellm/llms/anthropic/pass_through/messages/transformation.py`, 다른 제공자의 Messages 설정 | `crates/llms/src/base_llm/messages/`, `crates/llms/src/<provider>/messages/` | 정규화, 베타 정책, 도구, thinking, 시스템 메시지, 지원 파라미터 |
| Messages 데이터 구조 | `litellm/types/llms/anthropic_messages/` | `crates/llms-types/src/formats/messages/` | 허용 필드, 타입 구분자, 알 수 없는 필드, 직렬화, 사용량 |
| Responses 진입점과 호출 조율 | `litellm/responses/{dispatch,main}.py`, `litellm/rust_bridge/responses/` | `crates/python-bridge/src/routes/responses/`, `crates/inference-responses/src/` | create와 나머지 작업의 구분, 제공자 선택, HTTP와 WebSocket의 구분, 실행 허용 판단 |
| Responses 변환 | `litellm/llms/base_llm/responses/`, `litellm/llms/<provider>/responses/`, `litellm/completion_extras/litellm_responses_transformation/` | `crates/llms/src/base_llm/responses/`, `crates/llms/src/{openai,openrouter}/responses/` | 파라미터 재작성, 식별자, 도구, 에뮬레이션, 제공자 정책 |
| Responses 데이터 구조 | `litellm/types/llms/openai.py`, `litellm/types/responses/` | `crates/llms-types/src/formats/responses/` | 입출력 항목, 이벤트 구조, 추가 필드, 사용량 |
| 스트리밍과 복구 | Messages의 `streaming_iterator.py`, `fake_stream_iterator.py`, `agentic_streaming_iterator.py`, Responses의 `streaming_iterator.py`, `sse_output_recovery.py` | `crates/llms/src/base_llm/{messages,responses}/streaming.rs`, `crates/llms/src/base_llm/sse.rs`, `crates/framer/`, 추론 실행 드라이버 | 프레임 처리, 종료 이벤트, 오류, 버퍼링, 부분 출력, 취소, 자원 정리 |
| 제공자 등록과 모델 기능 정보 | `litellm/utils.py`, `litellm/litellm_core_utils/get_llm_provider_logic.py`, `litellm/llms/openai_like/`, 모델 카탈로그 | `crates/core-utils/`, `crates/inference/src/provider.rs`, 호출 준비 로직, Python 모델 기능 호스트 | 동적 제공자, 별칭, 모델 지원 기능, 인증 및 엔드포인트 선택 |
| 공통 요청 가공과 전송 | 프롬프트 템플릿, 타임아웃 결정 로직, 제공자 헤더, `litellm/llms/custom_httpx/`, 제공자 공통 유틸리티 | `crates/core-utils/`, `crates/http/`, 제공자 설정, Python 호스트 바인딩 | 공통 헬퍼 변경, 우선순위, 타임아웃, 요청 헤더, HTTP 오류 |
| 공개 오류와 메타데이터 | `litellm/exceptions.py`, 예외 매핑 유틸리티, `llm_response_utils/`, `litellm/anthropic_interface/`, `litellm/rust_bridge/failures.py` | 코어 오류 매핑, `crates/llms/src/error.rs`, 브리지 오류, 게이트웨이 오류 변환 | 예외 클래스, 상태 코드, 본문, 요청 ID, 응답 헤더, 내부 메타데이터 |
| 적용 정책과 위임 실행 | `litellm/rust_bridge/`, `litellm/utils.py`의 클라이언트 데코레이터 | `crates/python-bridge/`, `crates/host-python/`, `crates/callbacks-legacy-python/` | 실행 전 폴백, 실행 시작 후 폴백 금지, 호출자 태스크 유지, Python이 관리하는 상태 |
| 프록시와 라우팅 | `litellm/proxy/anthropic_endpoints/`, `litellm/proxy/response_api_endpoints/`, `litellm/router.py` | `crates/gateway-inference/`, `crates/router/`, `crates/router-types/` | HTTP 계약, 재시도, 배포 선택, 폴백, 예산 및 인증 연동 |
| 캐시와 콜백 및 비용 처리 | `litellm/caching/`, 로깅 유틸리티, 브리지 캐시와 기존 콜백 호스트 | `crates/cache-response/`, 추론 캐시, 브리지 캐시, 기존 콜백, 게이트웨이 의존 구성 요소 | 캐시 식별과 재생, 콜백 순서, 사용량, 비용 부과, 종료 결과 전달 |

safeguards, advisor 훅, MCP, 컨텍스트 관리, compaction, 파일 검색 에뮬레이션, 세션, 커스텀 도구, 도구 검색도 명시적인 점검 범위를 가져야 한다. 범위 검토를 완료하기 전에 각 기능이 네이티브 구현인지, Python에 위임되는지, 폴백하는지, 미지원인지, 의도적으로 다른지 확인해야 한다

### 소스 선택

우선 `litellm/` 아래의 모든 추적 파일, 관련 테스트, `litellm-rust/` 전체, 모델 카탈로그와 백업, 제공자가 관리하는 JSON 설정, 의존성 명세 및 잠금 파일에서 변경 후보를 넓게 수집한다. 이렇게 해야 새 폴더나 알려진 엔드포인트 경로 밖의 의존성도 발견할 수 있다. 파일 선택은 후보를 찾는 단계이고, 실제 관련성은 심볼과 호출 경로를 검토해 판단한다

Messages와 Responses의 직접 구현을 찾는 패턴에는 `litellm/messages/**`, `litellm/anthropic_interface/**`, `litellm/llms/anthropic/pass_through/**`, `litellm/llms/base_llm/anthropic_messages/**`, `litellm/types/llms/anthropic_messages/**`, `litellm/responses/**`, `litellm/types/responses/**`, `litellm/llms/base_llm/responses/**`, `litellm/llms/*/responses/**`, `litellm/llms/*/messages/**`를 포함한다

제공자별 디렉터리 구조를 고려해 `litellm/llms/azure_ai/anthropic/**`, `litellm/llms/bedrock/claude_platform/**`, `litellm/llms/vertex_ai/vertex_ai_partner_models/anthropic/**`, 제공자 공통 유틸리티, `litellm/llms/openai_like/**`도 포함한다. 브리지, 응답 어댑터, 프록시 엔드포인트 폴더, `litellm/main.py`, `litellm/__init__.py`, `litellm/utils.py`, `litellm/router.py`, 타입, 예외, 캐시, 공통 코어 유틸리티는 의존성으로 등록한다

특히 프롬프트 템플릿, 제공자 결정 로직, 헤더 추출, 예외 매핑, 로깅, 설정, 타임아웃, `litellm/anthropic_beta_headers_manager.py`, `litellm/anthropic_beta_headers_config.json`을 감시한다. 카탈로그 변경은 관련 모델 기능에 미치는 영향을 분류해야 하며, 무관한 모델 항목까지 자동으로 이식 대상으로 만들면 안 된다

매니페스트에서는 직접 구현, 공통 의존성, 테스트를 구분해 변경이 특정 범위에 영향을 주는 이유를 설명할 수 있게 한다. import 분석과 실행 추적은 빠진 의존성을 찾는 데 도움이 되지만, 동적 import, 제공자 등록, 문자열 이름을 통한 Python 호출 때문에 이것만으로 소스를 선택해서는 안 된다. 아직 분류되지 않은 후보 경로는 명시적으로 범위를 분류해야 하며, 조용히 통과시키지 않는다

## 체크포인트와 보고서 계약

`litellm-rust/python_parity.json` 같은 경로에 버전을 갖는 단일 매니페스트를 두고, `messages.sdk.anthropic.create.nonstream`, `responses.gateway.openai.create.sse`처럼 안정적인 식별자를 사용한다. 이 예시는 범위의 이름일 뿐, 현재 동작이 일치한다는 선언이 아니다

| 필드 | 의미와 요구 조건 |
| --- | --- |
| `id`, `surface`, `operation`, `provider`, `mode` | 검증하려는 범위를 정확히 정의한다. 필요한 경우 동기와 비동기 차이도 포함한다 |
| `python_paths`, `python_symbols`, `dependency_paths`, `rust_paths` | 소스와 구현을 선택하는 기준이다. 다대다 매핑을 허용한다 |
| `support`, `rollout`, `delegation` | 네이티브 지원 기능, 실행 허용 조건, 적용 정책, Python에 남겨 둔 책임을 각각 기록한다 |
| `parity_baseline_sha` | 명시적으로 승인한 차이를 제외하면 해당 범위의 미해결 동작 차이가 없는 지점이다. 기본 브랜치에 반영된 커밋의 전체 SHA이며, 아직 확정하지 않았다면 생략한다 |
| `reviewed_through_sha` | 별도의 검토 완료 지점이다. 미해결 작업이 있더라도 해당 작업을 모두 미해결 차이 목록에 유지하는 경우에만 앞으로 옮길 수 있다 |
| `rust_evidence_sha`, `evidence` | 체크포인트를 뒷받침하는 기존 구현의 불변 리비전과 동작 검증 근거다 |
| `gaps`, `intentional_differences` | 안정적인 ID, 영향받는 동작, 이유, 담당 역할, 근거, 존재하는 경우 후속 작업 링크를 기록한다 |

평가하지 않은 범위에는 동작 일치 기준점을 두지 않는다. 모든 항목을 현재 main SHA로 초기화해서는 안 된다. 각 범위의 초기 기준점은 상속된 동작과 기존 미구현 항목까지 포함해 검토한 뒤 확정한다. 검토를 마쳤더라도 부분 이식은 여전히 부분 이식이다

기준점을 앞으로 옮기려면 그사이의 모든 동작 변경에 대해 처리 상태가 확정되고 동작 검증 근거가 있어야 한다. 의도적 차이는 계속 명시하며, 관련 소스가 바뀌면 다시 검토한다. 폴백 덕분에 공개 SDK가 동작하더라도, 네이티브에서 지원하지 않는 기능은 미해결 이식 항목으로 남는다

초기 구현에서는 별도의 후속 체크포인트 커밋을 통해 기준점을 기본 브랜치에 이미 반영된 커밋으로 옮긴다. 같은 PR에서 Rust 구현을 보완했다면 해당 PR head의 검증 근거로 PR의 의무를 충족할 수 있지만, 기준점은 즉시 옮기지 않아도 된다. 이렇게 하면 커밋 안에 자신의 SHA를 넣는 문제와 squash 이전 커밋이 기본 브랜치의 이력에 남지 않는 문제를 피할 수 있다

검증 근거는 체크포인트보다 앞선 기존 구현 커밋을 참조할 수 있다. 테스트한 리비전에서 감시 대상 소스와 구현의 blob이 체크포인트 최종 커밋의 blob과 일치하는지 확인한다. 이후 Rust만 변경하더라도 Python 기준점이 그대로라는 이유로 검토를 생략해서는 안 된다

JSON 보고서와 짧은 Markdown 요약에는 기준점, 대상 브랜치, PR head, 구현, 정책, 매니페스트의 전체 리비전을 연결한다. 각 동작 변경에는 소스 경로와 심볼, diff hunk 또는 blob 참조, 영향받는 범위, 의존 관계의 이유, 외부에서 관찰할 수 있는 동작, 최종 구현 또는 폴백 경로, 처리 상태, 근거, 남은 차이를 기록한다

처리 상태는 `port_required`, `already_handled`, `shared_python`, `explicit_fallback`, `intentional_difference`, `no_runtime_effect`, `unresolved`를 사용한다. 에이전트가 분류를 제안하고 Rust 리뷰어가 수락하거나 거절한다. 규칙 기반 검사는 누락 여부와 리비전 연결을 확인하지만, 사람이나 모델이 내린 동작 분석의 결론까지 증명할 수는 없다

보고서에서는 기존의 미해결 동작 차이와 이번에 새로 생긴 작업을 구분한다. 분류한 범위와 동작 변경의 수를 집계하되, 아직 모르는 부분도 명시한다. 변경 줄 수나 일치하는 함수 비율로 기능의 동작 일치 수준을 측정하지 않는다

## 규칙 기반 변경 목록 검사와 에이전트 분석

먼저 불변 커밋 객체를 확정한다. 각 범위에서 `B`는 동작 일치 기준점이고 `T`는 대상 기본 브랜치의 SHA다. PR에서는 `H`를 제출된 head, `M`을 대상 브랜치와의 공통 조상인 merge base로 둔다. 객체 누락, 얕은 이력, 잘못된 조상 관계, 검토하지 않은 기준점은 명시적인 오류 또는 미평가 상태로 처리한다. 빈 변경 목록을 성공으로 반환해서는 안 된다

누적된 구현 차이는 두 커밋 사이의 최종 상태 diff로 확인하고, PR 자체의 변경은 merge base와 head 사이의 diff로 확인한다. 아래 명령은 변경 목록을 만드는 단계이며, 변수는 신뢰할 수 있는 이벤트 메타데이터와 검증된 매니페스트에서 가져온다

```sh
git cat-file -e "$B^{commit}"
git cat-file -e "$T^{commit}"
git merge-base --is-ancestor "$B" "$T"
git diff --no-ext-diff --no-textconv --find-renames=50% --name-status -z "$B" "$T"
M=$(git merge-base "$T" "$H")
git diff --no-ext-diff --no-textconv --find-renames=50% --name-status -z "$M" "$H"
```

선택 기준을 적용하기 전에 NUL로 구분된 전체 변경 목록을 파싱한다. 이전 이름과 새 이름 모두를 신뢰할 수 있는 기존 정책과 제안된 정책의 합집합에 대조한다. 추가, 삭제, 감시 범위 밖으로의 이름 변경, 범위 자체의 삭제가 계속 보여야 한다. PR이 자신의 감시 범위를 축소해 검사를 통과하게 해서는 안 된다

영향받는 범위마다 전체 소스 diff와 양쪽 스냅샷을 확인한 뒤 호출자, 의존성, 등록 정보, 실행 허용 판단, 폴백, 공개 응답 변환, 테스트를 검토한다. 커밋 이력은 중간 작업과 되돌리기를 이해하는 자료다. 현재 동작의 변화는 두 최종 상태 사이의 순수 diff로 판단한다

```sh
git log --format='%H %s' "$B..$T" -- <affected-paths>
git diff --no-ext-diff --no-textconv --find-renames=50% "$B" "$T" -- <affected-paths>
git show "$B:<path>"
git show "$T:<path>"
```

Rust와 브리지의 변경도 기록된 구현 리비전과 비교한다. Python 변경이 공통 Python 경로를 통해 이미 적용될 수도 있고, 기존 Rust 구현이 이미 처리할 수도 있으며, 같은 PR의 Rust 변경이 이를 보완할 수도 있다. 반대로 Rust만 바뀌어도 동작 일치가 깨질 수 있다

에이전트에는 검증된 변경 목록과 관련 스냅샷을 제공하되, 저장소의 텍스트는 신뢰할 수 없는 입력으로 취급한다. 에이전트는 바뀐 동작을 식별하고, 각 인터페이스의 최종 실행 경로를 추적하며, 해당 동작을 검증하는 최소한의 테스트를 제안하고, 지정된 리비전과 연결된 보고서를 만든다. 체크포인트를 스스로 옮기거나, 자신의 분석을 승인하거나, 근거가 없는데 성공으로 처리해서는 안 된다

`T`가 바뀌면 새 대상을 기준으로 누적 변경 목록을 다시 만들고, 실제로 대상 브랜치와 PR을 합친 병합 결과에서 동작 검증 근거를 확보한다. 바뀌지 않은 `H`만 테스트해서는 새로 반영된 변경과의 상호작용을 검증할 수 없다. merge queue를 사용한다면 실제 merge-group SHA를 사용한다. 이전 head나 정책 리비전에 대한 보고서로 현재 검사를 통과시켜서는 안 된다

## 기존 테스트와 검증 근거

기존 [동작 비교 하네스](../../tests/rust-python-harness/AGENTS.md)에는 SDK와 게이트웨이 인터페이스, 빠짐없는 전략별 분류, 요청과 공개 객체 및 오류와 스트림과 콜백 비교, 기록 및 재생 헬퍼가 정의되어 있다. 별도의 동작 비교 프레임워크를 만들기보다 이 개념과 픽스처를 재사용한다

현재 하네스 규칙은 로컬 실행만 허용하고 CI 통합을 명시적으로 금지한다. 조사한 스냅샷에서 `shared/unit_runners/contracts.py`에는 등록된 단위 테스트 계약이 없으며, `strategies/e2e_parity/__init__.py`에는 구현된 Messages 및 Responses 사례가 없다. `trace_parity`는 Rust span 대신 Python 호출을 수집한다. 이 자료는 점검 범위를 찾는 데 도움이 되지만 동작 일치 기준점을 확정하는 근거는 아니다. 첫 CI 변경 목록 검사기는 이 로컬 하네스와 독립적으로 둔다

[Rust 워크플로](../../.github/workflows/test-rust.yml)는 포맷 검사, Clippy, Cargo 테스트, 생성된 trace 검사, 새로 빌드한 wheel 테스트를 실행한다. 하지만 수동으로 지정한 Python 실행 조건 경로에는 대부분의 제공자 변환 코드와 `litellm/responses/streaming_iterator.py`가 빠져 있다. Rust CI가 통과했다는 사실만으로 관련 Python 변경을 모두 발견했다고 볼 수는 없다

동작 검증 근거에는 실제 명령, 종료 코드, 테스트 사례 식별자, 테스트한 리비전, wheel 또는 네이티브 빌드 식별 정보, 산출물 위치, 네이티브 분기가 실제로 실행되었다는 증거를 포함한다. 한쪽이 폴백하거나, 양쪽 모두 건너뛰거나, 바뀐 동작을 어느 쪽도 검증하지 않는다면 Python과 Rust 테스트가 모두 성공해도 충분하지 않다

요청 가공, 허용 필드, 오류, 수명 주기, 콜백 순서, 취소, 스트림 이벤트에 대해 재현 가능한 집중 회귀 테스트를 작성한다. 영향받는 SDK와 게이트웨이 인터페이스를 각각 검증한다. 실제 제공자나 실행 중인 프록시 확인이 필요한 변경에는 해당 검증을 수행하며, 건너뛴 항목이나 사용할 수 없는 인증 정보는 보고서에 명시한다. 테스트 명령을 고객 관점의 수정 증명으로 제시하지 않는다

## Rust 리뷰 필수화

[CODEOWNERS](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners)는 리뷰 요청을 연결하며, 저장소에서 코드 소유자 승인을 요구하면 나열된 소유자 중 한 명의 승인을 필수로 만들 수 있다. 같은 패턴에 여러 소유자를 적으면 그중 한 명이 승인하면 된다. 마지막으로 일치하는 패턴이 적용되므로, Python과 Rust 소유자를 겹치는 별도 행에 적어도 양쪽 승인을 모두 요구하게 되지는 않는다

조사한 시점의 [`.github/CODEOWNERS`](../../.github/CODEOWNERS)에는 비용 맵 JSON 파일 두 개만 등록되어 있다. [guard-main](https://github.com/BerriAI/litellm/rules/17360296)은 코드 소유자 승인을 요구하지만, 변경 후 기존 승인 무효화와 마지막 push 승인은 비활성화되어 있다. [main-merge](https://github.com/BerriAI/litellm/rules/16489521)는 승인 한 개와 squash 병합을 요구한다

감시 대상 Python 변경에 Rust 팀 승인 한 개를 필수로 요구하려면 GitHub 기본 기능인 [required reviewers](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/available-rules-for-rulesets#required-reviewers) 규칙을 사용한다. 이 저장소도 이미 [Protect Sensitive Files](https://github.com/BerriAI/litellm/rules/14572570)에서 `pull_request` 규칙의 파라미터 안에 `required_reviewers`를 사용한다. 해당 규칙은 `.github/**`와 일부 의존성 및 마이그레이션 파일에 팀 ID `16890674`의 승인을 요구하며, 이 팀이 `@BerriAI/litellm-engineering`임을 확인했다

[REST 규칙 스키마](https://docs.github.com/en/rest/repos/rules)의 필수 리뷰어 항목에는 `minimum_approvals`, `file_patterns`, `reviewer`를 사용한다. 이 항목 안에 숫자 `id`와 값이 `Team`인 `type`을 지정한다. 확인된 Rust 팀의 항목을 추가하고, Python 팀 승인도 필요하다면 같은 경로에 Python 팀 항목을 별도로 추가한다. 두 팀의 구성원이 겹치면 서로 다른 두 사람이 승인한다는 보장은 없다

조회 가능한 팀 목록에서는 Rust 마이그레이션 팀을 확인하지 못했다. 배포 전에 관리자가 해당 팀을 확인하거나 생성하고, 구성원과 저장소 쓰기 권한을 검증해야 한다. engineering 팀이 의도한 Python 소유자인지도 확인해야 한다. 임시 팀 이름이나 숫자 ID를 실제 설정에 사용해서는 안 된다

시범 적용 단계에서는 넓게 잡은 Python 후보 경로, 감시 대상 Rust 및 브리지 구현 경로, 관련 데이터 및 의존성 파일, 구현 차이 점검 정책과 검사기 및 보고서 계약과 워크플로 변경에 Rust 리뷰를 요구한다. 변경 목록이 범위 축소의 타당성을 뒷받침한 뒤에만 패턴을 좁힌다. 하나의 기준 선택 정의에서 코드 소유권 패턴과 ruleset 패턴을 생성하거나 검증하되, 각 시스템의 glob 해석 차이를 테스트한다

자동 리뷰 요청을 위해 CODEOWNERS를 추가한다면 넓은 패턴을 앞에, 좁은 패턴을 뒤에 둔다. 기존 비용 맵 소유자가 마지막으로 일치하는 항목에 유지되도록 한다. 같은 패턴의 소유자는 모두 한 줄에 적는다. CODEOWNERS 파일과 구현 차이 점검 인프라도 보호한다. 승인 강제는 GitHub의 팀별 필수 승인 규칙으로 처리한다

이후 push에서 바뀐 코드도 Rust 팀이 다시 승인하게 하려면 적용되는 PR 규칙에서 변경 후 기존 승인 무효화를 활성화해야 한다. 마지막 push 승인만 요구하면 다른 리뷰어가 새 변경을 승인하는 동안 이전 Rust 승인이 그대로 남을 수 있다. 적용 전에 대표 PR로 합쳐진 규칙의 실제 효과와 일반적인 우회 동작을 확인한다

## CI 경계와 적용 순서

모든 PR에서 가벼운 변경 목록 검사 작업을 실행하고, 후보가 없으면 영향이 없다는 결과를 명시적으로 반환한다. 워크플로 수준의 경로 필터는 필수 검사를 pending 상태로 남길 수 있다. PR 코드를 실행할 때는 읽기 전용 권한을 사용한다. 인증 정보를 사용하는 에이전트 작업은 신뢰할 수 있는 기준 브랜치 코드를 실행하고, 포크의 파일을 데이터로 취급해야 한다. PR이 제어하는 의존성을 설치하거나 코드를 실행해서는 안 된다. [비용 맵 검사](../../.github/workflows/cost-map-guard.yml)는 권한이 필요한 메타데이터 조회와 신뢰할 수 없는 변경을 분리하는 저장소 내 참고 사례다

게시하는 결과를 현재 PR head와 연결하고, 게시 직전에 head를 다시 확인한다. 새 push와 기준 브랜치 변경을 처리하며, merge queue를 도입하면 `merge_group`도 추가한다. 에이전트 출력이 없으면 검토 불가 상태이며, 동작 일치 통과가 아니다. 모델 분석은 참고 자료로 사용하고, 규칙 기반 게이트는 체크포인트 유효성, 분류 누락 여부, 미해결 차이 유지, 검증 근거의 리비전 일치를 확인하도록 한다

먼저 매니페스트 스키마, 소스 변경 목록 검사기, 보고서 스키마를 반영하고, 파일 추가와 삭제, 이름 변경, 공통 의존성, 새 범위, 잘못된 SHA, 오래된 보고서, 감시 경로 제거 시도에 대한 동작 테스트를 작성한다. 초기 상태는 미평가로 두고 보고서만 출력한다. 좁은 Messages 제공자 범위 하나와 Responses 인터페이스 하나를 점검한 뒤 초기 기준점을 선언한다

그다음 실제 Rust 팀을 확정하고 팀별 필수 승인 규칙과 CODEOWNERS 리뷰 요청을 설정한다. Python만 변경한 PR, Rust만 변경한 PR, 양쪽 구현을 함께 보완한 PR, 폴백 관련 PR, 매니페스트만 변경한 PR, 포크 PR을 검증한다. 시범 적용 후 규칙 기반 검사를 필수로 전환하고 범위를 점진적으로 넓힌다. 배포 단계에서는 실제 ruleset 설정과 현재 head의 봇 리뷰 및 CI 결과를 확인해야 하며, 누락되었거나 대기 중인 결과를 승인으로 취급해서는 안 된다

## 실제 변경을 이용한 점검 예시

[`a3251a3bfde17fb1c88896a1f2f3b7f9d2ef0345`](https://github.com/BerriAI/litellm/commit/a3251a3bfde17fb1c88896a1f2f3b7f9d2ef0345) 커밋은 Responses 스트리밍 실패 로그에 제공자의 요청 ID를 보존한다. 이 커밋은 `litellm/responses/streaming_iterator.py`, 공통 헬퍼인 `litellm/litellm_core_utils/llm_response_utils/get_headers.py`, Bedrock 이벤트 디코딩, 회귀 테스트를 변경하며 Rust 수정은 포함하지 않는다

```sh
git diff --no-ext-diff --no-textconv --name-status a3251a3bfde17fb1c88896a1f2f3b7f9d2ef0345^ a3251a3bfde17fb1c88896a1f2f3b7f9d2ef0345
git show a3251a3bfde17fb1c88896a1f2f3b7f9d2ef0345 -- litellm/responses/streaming_iterator.py litellm/litellm_core_utils/llm_response_utils/get_headers.py
```

외부에서 확인할 수 있는 변화는 스트림 실패 예외가 제공자의 요청 ID 헤더를 유지한다는 점이다. Responses 폴더만 감시하면 공통 헬퍼와 Bedrock 디코더를 놓친다. 모든 Python 변경에 Rust 수정을 요구해도 현재 SDK 경로를 잘못 분류하게 된다. 현재 Responses는 적용 정책에 따라 Python으로 직접 전달되므로 수정 사항이 그대로 적용된다

현재 SDK 결과는 디스패치와 실행 근거를 첨부해 Python이 담당하는 동작으로 기록하고, 네이티브 SDK 스트리밍은 미지원으로 남긴다. 네이티브 실행 허용 판단도 HTTP 스트리밍을 거부하지만, 이 폴백 경로를 입증하려면 적용 정책을 덮어쓰는 별도 테스트가 필요하다. 게이트웨이는 네이티브 Responses SSE를 사용하므로 오류 응답 변환과 실패 로깅을 따로 확인한다. Rust 수정이 없다는 사실은 검토할 작업 후보를 뜻하며, 게이트웨이 결함을 증명하지는 않는다

집중 재생 테스트에서는 제공자의 요청 ID 헤더가 있는 HTTP SSE 응답을 반환한 뒤 실패 이벤트를 보낸다. 각 인터페이스에 맞게 공개 오류, 상태 코드, 실패 메타데이터를 비교하고, 어느 구현이 실제로 실행되었는지 증명한다. 해당 실행 추적과 동작 검증 근거를 확보하기 전까지 게이트웨이의 처리 상태는 미해결로 유지한다
