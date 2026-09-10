# SSR API

This reference lists every export from `@pierre/trees/ssr`.

| Export                        | Kind     | Purpose                                                         |
| ----------------------------- | -------- | --------------------------------------------------------------- |
| `preloadFileTree`             | Function | Renders a tree to a `FileTreeSsrPayload`.                       |
| `serializeFileTreeSsrPayload` | Function | Creates declarative or DOM-inserted host markup from a payload. |
| `FileTreeSsrPayload`          | Type     | Holds the host start, shadow HTML, host end, and stable ID.     |

`preloadFileTree` and `serializeFileTreeSsrPayload` are also available from
`@pierre/trees`.
