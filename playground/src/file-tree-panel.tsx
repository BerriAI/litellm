import { useEffect, useRef } from 'react';

import { ExampleFileTree } from './file-tree';
import type { PlaygroundExample } from './types';

type FileTreePanelProps = {
  activePath?: string;
  example: PlaygroundExample;
  onOpenFile: (uri: string) => boolean;
};

export const FileTreePanel = ({ activePath, example, onOpenFile }: FileTreePanelProps) => {
  const container = useRef<HTMLElement>(null);
  const tree = useRef<ExampleFileTree | null>(null);
  const openFile = useRef(onOpenFile);
  openFile.current = onOpenFile;

  useEffect(() => {
    if (!container.current) {
      return;
    }
    const instance = new ExampleFileTree(container.current, uri => { openFile.current(uri); });
    instance.render(example, activePath);
    tree.current = instance;
    return () => {
      instance.cleanUp();
      tree.current = null;
    };
  }, [example]);

  useEffect(() => {
    if (activePath) {
      tree.current?.select(activePath);
    }
  }, [activePath]);

  return (
    <>
      <p className="tree-title files-title">FILES · {example.id}://</p>
      <nav ref={container} className="file-nav" aria-label={`${example.title} files`} />
    </>
  );
};
