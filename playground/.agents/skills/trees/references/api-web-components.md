# Web components API

Import `@pierre/trees/web-components` to register the `file-tree-container`
custom element. The entry exports these APIs:

| Export                      | Kind     | Purpose                                                                   |
| --------------------------- | -------- | ------------------------------------------------------------------------- |
| `FileTreeContainerLoaded`   | Value    | Confirms that the registration module ran.                                |
| `adoptDeclarativeShadowDom` | Function | Copies a declarative template into an empty shadow root.                  |
| `ensureFileTreeStyles`      | Function | Installs the core tree stylesheet in a shadow root.                       |
| `prepareFileTreeShadowRoot` | Function | Adopts server markup, installs styles, and measures the scrollbar gutter. |
