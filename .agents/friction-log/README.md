# Friction log

Friction hit while working in this repository, one directory per item:

```
<id>/
  friction.md   the write-up
  artifacts/    optional, whatever reproduces it
```

Reporting an entry gives it an owner. The write-up then carries an `issue:` link and mirrors what happens
to it. The whole directory is deleted once the friction is resolved. Every entry left here is still
outstanding, including friction in dependencies.

Do not maintain an index here. This directory is the index.

## Logging Friction

```sh
frog list    # what is already known
frog log     # add one
```

`frog log` writes the sections to fill in. Each id is when the friction was hit plus its title, so
the directory reads oldest-first.

Put anything that reproduces the friction in that entry's `artifacts/` and reference it from the
write-up. The next reader runs the reproduction instead of rebuilding it.

## For Agents

These rules live in `CLAUDE.md`, which `AGENTS.md` points at, so amend them there rather than duplicating them into `AGENTS.md`.

Managed by [Frog](https://github.com/wevm/frog).
