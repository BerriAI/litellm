---
name: trees
description:
  Use when an app uses @pierre/trees to render or control a file tree, including
  React, vanilla JavaScript, SSR, web components, selection, search, rename,
  drag and drop, icons, git status, and themes.
---

# `@pierre/trees`

Use `@pierre/trees` for an interactive file tree. Public state and callbacks use
path strings.

## Install

```bash
pnpm add @pierre/trees
```

Install `react` and `react-dom` when the app uses the React entry.

## Select an API reference

| Entry                          | Reference                                              |
| ------------------------------ | ------------------------------------------------------ |
| `@pierre/trees`                | [Core API](references/api-core.md)                     |
| `@pierre/trees/react`          | [React API](references/api-react.md)                   |
| `@pierre/trees/ssr`            | [SSR API](references/api-ssr.md)                       |
| `@pierre/trees/web-components` | [Web components API](references/api-web-components.md) |

## Select a recipe

| Task                                                    | Recipe                                                 |
| ------------------------------------------------------- | ------------------------------------------------------ |
| Render and update a tree in React                       | [Use React](references/recipe-react.md)                |
| Render and update a tree without React                  | [Use vanilla JavaScript](references/recipe-vanilla.md) |
| Preload a tree on the server                            | [Use SSR](references/recipe-ssr.md)                    |
| Apply a resolved Shiki or VS Code theme                 | [Apply a theme](references/recipe-theme.md)            |
| Configure search, rename, drag and drop, and git status | [Add interactions](references/recipe-interactions.md)  |
