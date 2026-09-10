const requireElement = <T extends HTMLElement>(id: string): T => {
  const element = document.getElementById(id);
  if (!element) {
    throw new Error(`Missing element: ${id}`);
  }
  return element as T;
};

export const elements = {
  activePath: requireElement<HTMLSpanElement>('active-path'),
  diagnosticStatus: requireElement<HTMLSpanElement>('diagnostic-status'),
  editorsParent: requireElement<HTMLDivElement>('editors'),
  exampleNav: requireElement<HTMLElement>('example-nav'),
  fileNav: requireElement<HTMLElement>('file-nav'),
  fileTreeTitle: requireElement<HTMLParagraphElement>('file-tree-title'),
  guidePath: requireElement<HTMLSpanElement>('guide-path'),
  lspStatus: requireElement<HTMLSpanElement>('lsp-status'),
  markdownEditorParent: requireElement<HTMLDivElement>('markdown-editor'),
  output: requireElement<HTMLPreElement>('output'),
  resetButton: requireElement<HTMLButtonElement>('reset-button'),
  revision: requireElement<HTMLSpanElement>('revision'),
  runButton: requireElement<HTMLButtonElement>('run-button'),
  runStatus: requireElement<HTMLSpanElement>('run-status'),
};
