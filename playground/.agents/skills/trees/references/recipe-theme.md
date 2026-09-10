# Recipe: apply a resolved theme

Convert one resolved Shiki or VS Code theme to host styles:

```tsx
import { themeToTreeStyles } from '@pierre/trees';
import { FileTree } from '@pierre/trees/react';

const treeStyle = {
  height: 320,
  ...themeToTreeStyles(resolvedTheme),
};

<FileTree model={model} style={treeStyle} />;
```

Recalculate the styles when the resolved theme changes. Set tree override CSS
properties on the same host style when the product needs a local color choice.
