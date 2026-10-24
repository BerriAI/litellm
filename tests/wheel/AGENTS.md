# Release wheel checks

- Run against a release-built wheel passed with `--wheel`
- Only run in the `rust-wheel` job, never select from a unit shard
- May invoke `readelf` and load the compiled native extension
- Keep facts in `.github/scripts/verify_linux_native_wheel.py` and assertions here
- Record build metrics with `ci_metric`, never write a separate summary
