# Recipe: use a file tree in vanilla JavaScript

Create the model and mount it in an element with a height:

```ts
import { FileTree } from '@pierre/trees';

const mount = document.querySelector<HTMLElement>('#files');
if (mount == null) throw new Error('Missing file tree mount');

mount.style.height = '320px';

const tree = new FileTree({
  paths: ['README.md', 'src/', 'src/index.ts'],
  initialExpansion: 'open',
  search: true,
});

tree.render({ containerWrapper: mount });
```

Use `add`, `remove`, `move`, or `resetPaths` to update paths. Call `cleanUp()`
when the host removes the tree.
