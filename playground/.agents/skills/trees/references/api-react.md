# React API

This reference lists every export from `@pierre/trees/react`.

| Export                     | Kind      | Purpose                                                                |
| -------------------------- | --------- | ---------------------------------------------------------------------- |
| `FileTree`                 | Component | Mounts a `FileTree` model in a React host element.                     |
| `FileTreeProps`            | Type      | Defines the model, header, context menu, preload data, and host props. |
| `FileTreePreloadedData`    | Type      | Selects the `id` and `shadowHtml` fields for hydration.                |
| `useFileTree`              | Hook      | Creates one stable `FileTree` model.                                   |
| `UseFileTreeResult`        | Type      | Holds the model returned by `useFileTree`.                             |
| `useFileTreeSelection`     | Hook      | Returns the selected path list and updates with the model.             |
| `useFileTreeSearch`        | Hook      | Returns search state and search actions.                               |
| `FileTreeSearchState`      | Type      | Defines the search snapshot and actions.                               |
| `useFileTreeSelector`      | Hook      | Subscribes to a selected part of model state.                          |
| `FileTreeSelector`         | Type      | Selects a value from a model.                                          |
| `FileTreeSelectorEquality` | Type      | Compares two selected values.                                          |

`FileTreeProps` extends React host attributes except `children`. Its specific
fields are:

| Field               | Purpose                                             |
| ------------------- | --------------------------------------------------- |
| `model`             | Supplies the required `FileTree` model.             |
| `header`            | Supplies React content for the header slot.         |
| `renderContextMenu` | Produces React content for the active context menu. |
| `preloadedData`     | Supplies server markup for hydration.               |
