# Recipe: preload a file tree on the server

Create one payload and pass it to the React tree:

```tsx
import { preloadFileTree } from '@pierre/trees/ssr';
import { FileTree, useFileTree } from '@pierre/trees/react';

const options = {
  id: 'project-files',
  paths: ['README.md', 'src/', 'src/index.ts'],
  initialExpansion: 'open' as const,
  initialVisibleRowCount: 8,
};

const preloadedData = preloadFileTree(options);

export function ProjectFiles() {
  const { model } = useFileTree(options);
  return (
    <FileTree
      model={model}
      preloadedData={preloadedData}
      style={{ height: 240 }}
    />
  );
}
```

For a direct HTML response, call `serializeFileTreeSsrPayload(payload)`. Pass
`dom` as the second argument when a DOM API inserts the complete markup string.
