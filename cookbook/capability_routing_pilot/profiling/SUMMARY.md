# Profiling discovery summary

The label pass completed for 220 candidates at a total cost of $0.109114. Counts were repository_debugging 158, feature_implementation 61, shell_build 0, algorithms_data 0, and unknown 1. Existing label records were reused; no model calls were made in this phase

The image inventory checked 80 of 220 candidates using manifest lookups only. It found 62 available images, 0 confirmed unavailable images, and 18 lookup errors. All 18 errors were manifest HEAD HTTP 401 responses and remain `check_error`, not unavailable. The remaining 140 candidates were not inspected

Per-category counts are:

| Category | Labeled | Checked | Available | Unavailable | Errors | Not inspected | Draft selected | Shortfall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| repository_debugging | 158 | 40 | 32 | 0 | 8 | 118 | 2 | 28 |
| feature_implementation | 61 | 40 | 30 | 0 | 10 | 21 | 0 | 30 |
| shell_build | 0 | 0 | 0 | 0 | 0 | 0 | 0 | unsupported |
| algorithms_data | 0 | 0 | 0 | 0 | 0 | 0 | 0 | unsupported |
| unknown | 1 | 0 | 0 | 0 | 0 | 1 | 0 | excluded |

The draft selects only the resolved deterministic hash-order prefix. The repository-debugging IDs are `pydicom__pydicom-1413` and `pydicom__pydicom-1192`. Feature implementation has no eligible IDs because its first candidate has an unresolved 401 check. The 18 error rows are listed as exclusions in the manifest. This is an incomplete draft, not a frozen cohort. The errors ahead of the cutoff still need one bounded retry before freezing

No 429 response was recorded, so no `Retry-After` backoff was used. The collector did not persist `RateLimit-Remaining`; its exact value is unavailable. No credentials were used or recorded

Across the 60 confirmed available image candidates, compressed layers total 86,932,156,867 bytes (80.962 GiB). The peak planned three-image batch is 6,264,019,473 bytes (5.834 GiB). These are discovery estimates, not the current two-task prefix-eligible draft. Disk free space was 78 GiB before held-out image cleanup and 79 GiB after cleanup, with 79 GiB observed during discovery. No profiling images were pulled

The earlier 15-run durations were 187.882 seconds mean and 272.907 seconds p90. For 180 solver runs across three workers, the corresponding estimates are 11,272.907 seconds (3.13 hours) at the mean and 16,374.392 seconds (4.55 hours) at p90, before validation time. Estimated spend is $67.86 at the prior $0.377/run mean and up to $180 at the $1/run cap, plus the completed $0.109114 label spend

Focused offline tests passed: 18 passed in 5.66 seconds. No solver, Jev, smoke, gold, or noop execution was launched. The profiling key alias is `routing-poc-profiling-2026-10-21`, with a $5 maximum budget
