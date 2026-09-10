# Recipe: use a file tree in React

Create the model once and pass it to the component:

```tsx
'use client';

import { FileTree, useFileTree } from '@pierre/trees/react';

export function ProjectFiles({ paths }: { paths: readonly string[] }) {
  const { model } = useFileTree({
    paths,
    initialExpansion: 'open',
    search: true,
  });

  return <FileTree model={model} style={{ height: 320 }} />;
}
```

Call model methods for updates after model creation. For example, call
`model.resetPaths(paths)` after the source path list changes.
