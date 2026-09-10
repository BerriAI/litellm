# Core API

This reference lists every export from `@pierre/trees` and every public
`FileTree` member.

## Contents

- [Runtime values](#runtime-values)
- [`FileTree` members](#filetree-members)
- [Configuration and state types](#configuration-and-state-types)
- [Mutation types](#mutation-types)
- [Interaction types](#interaction-types)
- [Presentation types](#presentation-types)
- [Constants](#constants)

## Runtime values

| Export                          | Kind     | Purpose                                                           |
| ------------------------------- | -------- | ----------------------------------------------------------------- |
| `FileTree`                      | Class    | Owns the tree model, renders it, and exposes path-based controls. |
| `prepareFileTreeInput`          | Function | Prepares and optionally sorts a path list for reuse.              |
| `preparePresortedFileTreeInput` | Function | Prepares a path list that already has final order.                |
| `preloadFileTree`               | Function | Creates server-rendered tree markup and hydration data.           |
| `serializeFileTreeSsrPayload`   | Function | Joins an SSR payload into one host markup string.                 |
| `themeToTreeStyles`             | Function | Maps a resolved theme to tree host CSS properties.                |
| `getBuiltInSpriteSheet`         | Function | Gets the SVG sprite for a built-in icon set.                      |
| `createFileTreeIconResolver`    | Function | Creates an icon resolver from an icon configuration.              |

## `FileTree` members

| Member                           | Purpose                                                     |
| -------------------------------- | ----------------------------------------------------------- |
| `new FileTree(options)`          | Creates a model from paths or prepared input.               |
| `FileTree.LoadedCustomComponent` | Reports whether the file-tree custom element module loaded. |
| `render(props)`                  | Mounts the tree into a wrapper or host element.             |
| `hydrate(props)`                 | Attaches the model to preloaded host markup.                |
| `unmount()`                      | Removes the mounted view and keeps the model.               |
| `cleanUp()`                      | Removes the view and destroys model resources.              |
| `getFileTreeContainer()`         | Gets the mounted host element.                              |
| `getItem(path)`                  | Gets a path handle or `null`.                               |
| `getFocusedItem()`               | Gets the focused item handle or `null`.                     |
| `getFocusedPath()`               | Gets the focused path or `null`.                            |
| `getSelectedPaths()`             | Gets the selected paths.                                    |
| `getComposition()`               | Gets the current header and context-menu configuration.     |
| `getItemHeight()`                | Gets the resolved row height.                               |
| `getDensityFactor()`             | Gets the resolved density factor.                           |
| `subscribe(listener)`            | Subscribes to model changes.                                |
| `focusPath(path)`                | Focuses a path.                                             |
| `focusNearestPath(path)`         | Focuses and returns the nearest available path.             |
| `scrollToPath(path, options?)`   | Scrolls a path into view.                                   |
| `add(path)`                      | Adds one path.                                              |
| `remove(path, options?)`         | Removes one path.                                           |
| `move(from, to, options?)`       | Moves one path.                                             |
| `batch(operations)`              | Applies several path mutations together.                    |
| `resetPaths(paths, options?)`    | Replaces the full path set.                                 |
| `onMutation(type, handler)`      | Subscribes to mutation events.                              |
| `setSearch(value)`               | Sets or clears the search query.                            |
| `openSearch(initialValue?)`      | Opens the search session.                                   |
| `closeSearch()`                  | Closes the search session.                                  |
| `isSearchOpen()`                 | Tests whether search is open.                               |
| `getSearchValue()`               | Gets the search query.                                      |
| `getSearchMatchingPaths()`       | Gets paths that match the query.                            |
| `focusNextSearchMatch()`         | Focuses the next search match.                              |
| `focusPreviousSearchMatch()`     | Focuses the previous search match.                          |
| `startRenaming(path?, options?)` | Starts inline rename and reports whether it started.        |
| `setGitStatus(status?)`          | Replaces all git status entries.                            |
| `applyGitStatusPatch(patch)`     | Applies a partial git status update.                        |
| `setIcons(icons?)`               | Replaces the icon configuration.                            |
| `setComposition(composition?)`   | Replaces header and context-menu configuration.             |

## Configuration and state types

| Export                     | Purpose                                                         |
| -------------------------- | --------------------------------------------------------------- |
| `FileTreeOptions`          | Defines input, behavior, rendering, and presentation options.   |
| `FileTreePreparedInput`    | Holds a reusable prepared path list.                            |
| `FileTreeInitialExpansion` | Selects closed, open, or depth-based initial expansion.         |
| `FileTreeSortComparator`   | Compares two path entries.                                      |
| `FileTreeSortEntry`        | Describes one path for a sort comparator.                       |
| `FileTreeRenderOptions`    | Configures row height, row count, overscan, and sticky folders. |
| `FileTreeRenderProps`      | Selects the wrapper or existing host for `render`.              |
| `FileTreeHydrationProps`   | Selects the host for `hydrate`.                                 |
| `FileTreeVisibleRow`       | Describes one visible tree row.                                 |
| `FileTreeItemHandle`       | Represents a file or directory item.                            |
| `FileTreeFileHandle`       | Controls one file item.                                         |
| `FileTreeDirectoryHandle`  | Controls one directory item and its expansion.                  |
| `FileTreeListener`         | Defines a model subscription callback.                          |
| `FileTreeSsrPayload`       | Holds the host and shadow markup for server output.             |

## Mutation types

| Export                              | Purpose                                                        |
| ----------------------------------- | -------------------------------------------------------------- |
| `FileTreeMutationHandle`            | Defines the public path mutation methods.                      |
| `FileTreeBatchOperation`            | Describes one add, remove, or move in a batch.                 |
| `FileTreeCollisionStrategy`         | Selects error, replace, or skip behavior for a move collision. |
| `FileTreeMoveOptions`               | Configures move collision behavior.                            |
| `FileTreeRemoveOptions`             | Configures recursive removal.                                  |
| `FileTreeResetOptions`              | Configures a path reset and optional prepared input.           |
| `FileTreeResetPreparedOptions`      | Configures a reset that uses prepared input.                   |
| `FileTreeMutationEvent`             | Represents any mutation event.                                 |
| `FileTreeMutationSemanticEvent`     | Represents an add, remove, move, or reset event.               |
| `FileTreeMutationEventType`         | Names a mutation operation.                                    |
| `FileTreeMutationEventForType`      | Selects the event shape for an operation name.                 |
| `FileTreeMutationEventInvalidation` | Describes the state invalidation from a mutation.              |
| `FileTreeAddEvent`                  | Describes one add result.                                      |
| `FileTreeRemoveEvent`               | Describes one remove result.                                   |
| `FileTreeMoveEvent`                 | Describes one move result.                                     |
| `FileTreeResetEvent`                | Describes one reset result.                                    |
| `FileTreeBatchEvent`                | Describes one batch result.                                    |

## Interaction types

| Export                            | Purpose                                            |
| --------------------------------- | -------------------------------------------------- |
| `FileTreeSelectionChangeListener` | Receives the selected path list.                   |
| `FileTreeSearchChangeListener`    | Receives the current search query.                 |
| `FileTreeSearchSessionHandle`     | Defines search session methods.                    |
| `FileTreeSearchMode`              | Selects how nonmatching rows appear.               |
| `FileTreeSearchBlurBehavior`      | Selects search behavior after focus leaves.        |
| `FileTreeScrollOffset`            | Selects top, center, or nearest scroll alignment.  |
| `FileTreeScrollToPathOptions`     | Configures focus and alignment for `scrollToPath`. |
| `FileTreeDragAndDropConfig`       | Configures drag rules and completion callbacks.    |
| `FileTreeDropTarget`              | Describes the current drop target.                 |
| `FileTreeDropContext`             | Describes dragged paths and their target.          |
| `FileTreeDropResult`              | Describes the completed move or batch operation.   |
| `FileTreeRenamingConfig`          | Configures rename rules and callbacks.             |
| `FileTreeRenamingItem`            | Describes the item offered to a rename rule.       |
| `FileTreeRenameEvent`             | Describes a completed rename.                      |

## Presentation types

| Export                             | Purpose                                                     |
| ---------------------------------- | ----------------------------------------------------------- |
| `FileTreeCompositionOptions`       | Configures header and context-menu composition.             |
| `FileTreeHeaderCompositionOptions` | Supplies header HTML or a header renderer.                  |
| `ContextMenuItem`                  | Describes the file or directory for a context menu.         |
| `ContextMenuOpenContext`           | Supplies menu position, close, and focus controls.          |
| `ContextMenuAnchorRect`            | Describes the menu anchor rectangle.                        |
| `ContextMenuTriggerMode`           | Selects right-click, button, or both triggers.              |
| `ContextMenuButtonVisibility`      | Selects when the row menu button appears.                   |
| `FileTreeRowDecoration`            | Describes text or icon content in the decoration lane.      |
| `FileTreeRowDecorationContext`     | Supplies the item and visible row to a decoration renderer. |
| `FileTreeRowDecorationRenderer`    | Produces one row decoration.                                |
| `GitStatus`                        | Names a supported git status.                               |
| `GitStatusEntry`                   | Assigns a git status to one path.                           |
| `FileTreeGitStatusPatch`           | Adds, changes, or removes git status entries.               |
| `FileTreeBuiltInIconSet`           | Names a built-in icon set.                                  |
| `FileTreeIconConfig`               | Configures built-in and custom icon rules.                  |
| `FileTreeIcons`                    | Accepts an icon set name or icon configuration.             |
| `RemappedIcon`                     | Names or defines a replacement SVG symbol.                  |
| `FileTreeDensity`                  | Accepts a density keyword or numeric factor.                |
| `FileTreeDensityKeyword`           | Names compact, default, or relaxed density.                 |
| `FileTreeDensityPreset`            | Holds a density factor and row height.                      |
| `TreeThemeInput`                   | Defines the resolved theme accepted by `themeToTreeStyles`. |
| `TreeThemeStyles`                  | Maps tree CSS property names to values.                     |

## Constants

| Export                                         | Purpose                                                         |
| ---------------------------------------------- | --------------------------------------------------------------- |
| `FILE_TREE_TAG_NAME`                           | Provides the `file-tree-container` element name.                |
| `FILE_TREE_STYLE_ATTRIBUTE`                    | Provides the core style marker attribute.                       |
| `FILE_TREE_UNSAFE_CSS_ATTRIBUTE`               | Provides the custom style marker attribute.                     |
| `FILE_TREE_SCROLLBAR_MEASURE_ATTRIBUTE`        | Provides the scrollbar measurement attribute.                   |
| `FILE_TREE_SCROLLBAR_GUTTER_STYLE_ATTRIBUTE`   | Provides the measured scrollbar style attribute.                |
| `FILE_TREE_SCROLLBAR_GUTTER_MEASURED_PROPERTY` | Provides the measured scrollbar CSS property.                   |
| `FILE_TREE_DEFAULT_ITEM_HEIGHT`                | Provides the default row height.                                |
| `FILE_TREE_DENSITY_PRESETS`                    | Maps each density keyword to its preset.                        |
| `FLATTENED_PREFIX`                             | Provides the identifier prefix for a flattened directory chain. |
| `HEADER_SLOT_NAME`                             | Provides the header slot name.                                  |
| `CONTEXT_MENU_SLOT_NAME`                       | Provides the context-menu slot name.                            |
| `CONTEXT_MENU_TRIGGER_TYPE`                    | Provides the context-menu trigger type.                         |
