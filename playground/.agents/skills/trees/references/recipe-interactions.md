# Recipe: add file tree interactions

Enable only the interactions that the product exposes:

```ts
const tree = new FileTree({
  paths,
  search: true,
  renaming: {
    onRename(event) {
      renamePath(event.sourcePath, event.destinationPath);
    },
  },
  dragAndDrop: {
    canDrop({ target }) {
      return target.kind === 'directory';
    },
    onDropComplete(event) {
      saveMove(event);
    },
  },
  gitStatus,
});
```

Use `openSearch()` to open search from an application command. Use
`startRenaming(path)` to start rename from a menu. Use `setGitStatus()` or
`applyGitStatusPatch()` after repository state changes.

Directory input paths end with `/`. File input paths do not end with `/`.
