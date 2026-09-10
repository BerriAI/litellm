import { FileTree } from '@pierre/trees';

import type { PlaygroundExample } from './types';

export class ExampleFileTree {
  private tree: FileTree | null = null;
  private syncingSelection = false;

  public constructor(
    private readonly container: HTMLElement,
    private readonly openFile: (uri: string) => void,
  ) {}

  public render(example: PlaygroundExample, selectedPath?: string) {
    this.tree?.cleanUp();
    this.container.replaceChildren();
    this.container.setAttribute('aria-label', `${example.title} files`);

    const directories = new Set<string>();
    example.files.forEach(file => {
      const segments = file.path.split('/');
      for (let index = 1; index < segments.length; index += 1) {
        directories.add(`${segments.slice(0, index).join('/')}/`);
      }
    });

    this.tree = new FileTree({
      paths: [...directories, ...example.files.map(file => file.path)],
      initialExpansion: 'open',
      initialSelectedPaths: selectedPath ? [selectedPath] : [],
      icons: { set: 'complete', colored: true },
      density: 'compact',
      stickyFolders: true,
      onSelectionChange: selectedPaths => {
        if (this.syncingSelection) {
          return;
        }
        const selectedFile = example.files.find(file => file.path === selectedPaths.at(-1));
        if (selectedFile) {
          this.openFile(selectedFile.uri);
        }
      },
    });
    this.tree.render({ containerWrapper: this.container });
  }

  public select(path: string) {
    this.syncingSelection = true;
    this.tree?.getSelectedPaths().forEach(selectedPath => {
      if (selectedPath !== path) {
        this.tree?.getItem(selectedPath)?.deselect();
      }
    });
    this.tree?.getItem(path)?.select();
    this.tree?.scrollToPath(path, { focus: false, offset: 'nearest' });
    this.syncingSelection = false;
  }

  public cleanUp() {
    this.tree?.cleanUp();
    this.tree = null;
  }
}
