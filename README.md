PR #43723 local screenshots

Merge base: 657bb777fa557f7089c56fedb42f3056d7b55847
Unfixed rebased PR: a4492b9b3c85cf201d39c0a01435f7b5c9c944e8
Final source: 1cdc43665c130af8fe0f74e3a453c4381e20b5dc

Headless Chromium drove the dashboard at http://localhost:3000/agents/ against source proxy localhost:4019 and isolated PostgreSQL. All UUIDs shown are synthetic test inputs. Registration is demonstrated, not successful Entra authentication. One deliberately aborted tenant-list request demonstrates provider recovery; the succeeding request hits the actual proxy. See PR body for exact steps. Red boxes annotate actual screenshots.
