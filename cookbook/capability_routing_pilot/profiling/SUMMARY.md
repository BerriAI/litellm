# Profiling discovery summary

The label pass completed for 220 candidates at a total cost of $0.109114. Counts were repository_debugging 158, feature_implementation 61, shell_build 0, algorithms_data 0, and unknown 1. The runner and manifest-guard work made no additional model calls

The image inventory checked 80 of 220 candidates using manifest lookups only. It found 62 accessible images, 0 confirmed unavailable images, and 18 HTTP 401 access errors. Each error received one fresh anonymous token scoped to its repository and one HEAD retry. All 18 retries still returned 401, and each decoded token had an empty repository access claim. A single `docker manifest inspect` on one candidate also returned unauthorized. An empty claim can mean a missing or private repository, so absence is not confirmed. These candidates are recorded as `access_error` exclusions, not unavailable. No further discovery was needed because both supported categories already had at least 30 accessible images

Per-category counts are:

| Category | Labeled | Checked | Available | Unavailable | Errors | Not inspected | Frozen selected | Shortfall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| repository_debugging | 158 | 40 | 32 | 0 | 8 | 118 | 30 | 0 |
| feature_implementation | 61 | 40 | 30 | 0 | 10 | 21 | 30 | 0 |
| shell_build | 0 | 0 | 0 | 0 | 0 | 0 | 0 | unsupported |
| algorithms_data | 0 | 0 | 0 | 0 | 0 | 0 | 0 | unsupported |
| unknown | 1 | 0 | 0 | 0 | 0 | 1 | 0 | excluded |

The frozen manifest selects the first 30 accessible images in each supported category, ordered by `sha256(seed + task_id)`. It was frozen at `2026-10-21T03:24:04.314265+00:00` with SHA-256 `15720dee12d40b62780837ed12a403f3951e9c5bb1ed18ec0d5d5d1f4776bff4`. The 18 `access_error` exclusions and retry results are included in the manifest. Individual timestamps for the original successful checks were not persisted, so the freeze timestamp records when the available inventory was finalized

No 429 response was recorded. `RateLimit-Remaining` and `Retry-After` were absent from all 18 retry responses and are persisted as null. No credentials were used or recorded

Across the 60 selected image candidates, compressed layers total 86,932,156,867 bytes (80.962 GiB). The peak planned three-image batch is 6,264,019,473 bytes (5.834 GiB). Disk free space was 78 GiB before held-out image cleanup and 79 GiB after cleanup, with 79 GiB observed during discovery. No profiling images were pulled

The earlier 15-run durations were 187.882 seconds mean and 272.907 seconds p90. For 180 solver runs across three workers, the corresponding estimates are 11,272.907 seconds (3.13 hours) at the mean and 16,374.392 seconds (4.55 hours) at p90, before validation time. Estimated spend is $67.86 at the prior $0.377/run mean and up to $180 at the $1/run cap, plus the completed $0.109114 label spend

No solver, Jev, smoke, gold, or noop execution was launched. The profiling key alias is `routing-poc-profiling-2026-10-21`, with a $5 maximum budget
