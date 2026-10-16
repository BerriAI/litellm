# Python과 Rust 변경 목록 확인

저장소 루트에서 실행한다. 이 명령은 Python과 Rust 경로 매핑을 사용해 두 커밋 사이의 변경 목록과 파일별 diff를 만든다. 동작이 같은지 판정하거나 테스트를 실행하지 않으며, CI에는 연결하지 않는다

```sh
python -m tests.rust-python-harness drift --target HEAD
```

기본 매니페스트는 이 폴더의 [drift.json](drift.json)이다. `checkpoint_sha`는 모든 매핑에 공통으로 적용하는 단일 전역 비교 스냅샷이다. 그 커밋을 검토했거나 Python과 Rust가 일치한다는 뜻은 아니다. 초기 기준은 기본 브랜치에 반영된 `6530b329f43ef456c502ba2db7656f871d62474e`이며, 해당 시점의 일치 여부는 `checkpoint_state`의 `unknown`으로 기록한다

| `checkpoint_state` | 기준 커밋에서 기록한 상태 |
| --- | --- |
| `unknown` | 동작 일치 여부를 아직 확인하지 않았다 |
| `drifted` | Python과 Rust 사이의 구현 차이를 확인했다 |
| `aligned` | 정한 매핑 범위에서 동작을 확인하고 필요한 수정을 마쳤다 |

SHA와 상태는 별도로 기록한다. 차이가 있는 현재 소스를 먼저 기준으로 잡아도 된다. 이후 차이를 수정하고 확인했을 때 새 기준 커밋과 `aligned`를 기록할 수 있다. `aligned`는 기준 커밋의 기록이며 이후 대상 커밋의 일치를 자동으로 보증하지 않는다. 매핑은 초기 점검 경계이므로 전체 Python 기능의 이식을 뜻하지 않는다

이번 실행의 비교 기준을 명시하려면 `--since`를 지정한다

```sh
python -m tests.rust-python-harness drift --since 6530b329f43ef456c502ba2db7656f871d62474e --target HEAD
```

`--since`는 이번 실행의 비교 기준만 바꾼다. 보고서의 `comparison_sha`에 실제 기준 커밋의 전체 SHA가 기록되고 `comparison_source`는 `override`가 된다. `checkpoint_sha`와 `checkpoint_state`는 바꾸지 않는다

체크포인트가 없는 매니페스트도 사용할 수 있다. `checkpoint_sha`가 `null`이고 상태가 `unknown`이면 탐색용 diff가 있어도 보고서 상태는 `unreviewed`다. 저장된 체크포인트보다 과거의 커밋을 대상으로 확인하려면 별도 매니페스트에서 이 조합을 사용하고 `--since`와 `--target`을 지정한다. 지정한 비교 기준은 대상 커밋의 조상이어야 한다

원격 브랜치를 비교하려면 호출자가 먼저 가져와야 한다. 명령 자체는 fetch나 pull을 하지 않는다

```sh
git fetch origin
python -m tests.rust-python-harness drift --target origin/main --output /tmp/litellm-drift.json
```

다른 매핑 파일은 `--manifest PATH`로 선택한다. `--output PATH`를 생략하면 JSON 보고서를 표준 출력으로 확인할 수 있다. 보고서를 정상 생성하면 종료 코드는 0이고, 입력이나 Git 처리 오류로 생성하지 못하면 1이다. 종료 코드 0은 동작 일치나 검토 완료를 뜻하지 않는다

보고서에는 비교에 사용한 전체 커밋 SHA, 기록된 체크포인트 상태, 변경 파일의 patch와 해당 매핑 ID, 매핑되지 않은 후보 변경, 현재 작업 디렉터리의 미커밋 변경이 담긴다. 보고서의 `mappings`에는 비교 양 끝과 현재 매니페스트에서 합친 경로 매핑이 담기므로 관련 Rust 경로도 확인할 수 있다. 하나의 파일이 여러 매핑에 속할 수 있다. 비교 결과는 커밋 스냅샷 기준이며, dirty 항목의 변경 내용은 커밋 간 diff에 포함되지 않는다

| 보고서 `status` | 의미 |
| --- | --- |
| `unreviewed` | 저장된 비교 스냅샷이 없다 |
| `changes_found` | 비교 범위에서 후보 변경을 찾았다 |
| `no_changes` | 비교 범위에 후보 변경이 없다. `aligned`나 검토 완료를 뜻하지 않는다 |

보고서 상태는 `checkpoint_state`와 별개다. 예를 들어 기준 상태가 `drifted`이고 이후 변경이 없어도 `no_changes`가 될 수 있다

매니페스트의 `candidate_paths`는 Python, Rust, 테스트, 모델 데이터와 의존성 변경을 넓게 수집한다. `mappings`는 현재 알고 있는 Messages, Responses 및 공통 실행 경로를 연결한다. 목록에 포함됐다는 이유로 해당 기능이 Rust에 구현됐다고 해석해서는 안 된다

경로는 저장소 기준의 정확한 파일명이나 `/`로 끝나는 디렉터리 접두사다. glob은 지원하지 않는다. 새 제공자나 공통 의존성이 생기면 미분류 변경을 검토하고 필요한 매핑을 추가한다. 테스트와 의존성처럼 특정 기능으로 바로 분류하기 어려운 파일은 후보로 남겨 두고 diff를 확인한다

기준을 옮길 때는 기본 브랜치에 이미 반영된 커밋의 전체 SHA와 그 시점의 상태를 사람이 함께 기록하고 별도 변경으로 커밋한다. 상태를 확인하지 않았다면 `unknown`, 차이를 확인했다면 `drifted`로 남긴다. `aligned`는 관련 동작과 구현을 확인한 뒤에만 기록한다. 명령은 두 필드를 자동으로 변경하지 않는다

더 넓은 점검 범위와 이후 검토 체계는 [설계 문서](../../litellm-rust/docs/python_rust_drift.md)에 정리돼 있다
