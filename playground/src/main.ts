import { markdown } from '@codemirror/lang-markdown';
import { rust } from '@codemirror/lang-rust';
import {
  languageServerExtensions,
  jumpToDefinition,
  LSPClient,
  LSPPlugin,
  Workspace,
  type Transport,
  type WorkspaceFile,
} from '@codemirror/lsp-client';
import { oneDark } from '@codemirror/theme-one-dark';
import { Decoration, MatchDecorator, ViewPlugin } from '@codemirror/view';
import { EditorView, basicSetup } from 'codemirror';

import './style.css';

const STORAGE_KEY = 'litellm-rust-playground-files';

type PlaygroundFile = {
  languageId: 'rust' | 'toml';
  path: string;
  source: string;
  target: string;
  uri: string;
};

type PlaygroundGuide = {
  path: string;
  source: string;
  target: string;
};

type PlaygroundExample = {
  files: PlaygroundFile[];
  guide: PlaygroundGuide;
  id: string;
  title: string;
};

type PlaygroundInfo = {
  examples: PlaygroundExample[];
  rootUri: string;
  revision: string;
};

type RunResult = {
  success: boolean;
  output: string;
};

type StoredFiles = Record<string, string>;

class OpenFile implements WorkspaceFile {
  public version = 0;
  public doc;

  public constructor(
    public readonly uri: string,
    public readonly languageId: string,
    public readonly view: EditorView,
  ) {
    this.doc = view.state.doc;
  }

  public getView() {
    return this.view;
  }
}

class PlaygroundWorkspace extends Workspace {
  public files: OpenFile[] = [];

  public constructor(
    client: LSPClient,
    private readonly showFile: (uri: string) => EditorView | null,
  ) {
    super(client);
  }

  public syncFiles() {
    return this.files.flatMap(file => {
      const plugin = LSPPlugin.get(file.view);
      if (!plugin || plugin.unsyncedChanges.empty) {
        return [];
      }
      const changes = plugin.unsyncedChanges;
      const prevDoc = file.doc;
      file.doc = file.view.state.doc;
      file.version += 1;
      plugin.clear();
      return [{ file, prevDoc, changes }];
    });
  }

  public openFile(uri: string, languageId: string, view: EditorView) {
    const file = new OpenFile(uri, languageId, view);
    this.files = [...this.files, file];
    this.client.didOpen(file);
  }

  public closeFile(uri: string) {
    this.files = this.files.filter(file => file.uri !== uri);
    this.client.didClose(uri);
  }

  public displayFile(uri: string) {
    return Promise.resolve(this.showFile(uri));
  }
}

const requireElement = <T extends HTMLElement>(id: string): T => {
  const element = document.getElementById(id);
  if (!element) {
    throw new Error(`Missing element: ${id}`);
  }
  return element as T;
};

const lspStatus = requireElement<HTMLSpanElement>('lsp-status');
const diagnosticStatus = requireElement<HTMLSpanElement>('diagnostic-status');
const revision = requireElement<HTMLSpanElement>('revision');
const runButton = requireElement<HTMLButtonElement>('run-button');
const resetButton = requireElement<HTMLButtonElement>('reset-button');
const runStatus = requireElement<HTMLSpanElement>('run-status');
const output = requireElement<HTMLPreElement>('output');
const editorsParent = requireElement<HTMLDivElement>('editors');
const markdownEditorParent = requireElement<HTMLDivElement>('markdown-editor');
const exampleNav = requireElement<HTMLElement>('example-nav');
const fileNav = requireElement<HTMLElement>('file-nav');
const activePath = requireElement<HTMLSpanElement>('active-path');
const guidePath = requireElement<HTMLSpanElement>('guide-path');

const commandClickDefinition = EditorView.domEventHandlers({
  mousedown(event, view) {
    if ((!event.metaKey && !event.ctrlKey) || event.button !== 0) {
      return false;
    }
    const position = view.posAtCoords({ x: event.clientX, y: event.clientY });
    if (position === null) {
      return false;
    }
    view.dispatch({ selection: { anchor: position } });
    const handled = jumpToDefinition(view);
    if (handled) {
      event.preventDefault();
    }
    return handled;
  },
});

const playgroundLinkMatcher = new MatchDecorator({
  regexp: /\[[^\]\n]+\]\([a-z][a-z0-9+.-]*:\/\/[^)\s]+\)/gi,
  decoration: Decoration.mark({ class: 'cm-playground-link' }),
});

const playgroundLinkDecorations = ViewPlugin.fromClass(
  class {
    public decorations;

    public constructor(view: EditorView) {
      this.decorations = playgroundLinkMatcher.createDeco(view);
    }

    public update(update: Parameters<typeof playgroundLinkMatcher.updateDeco>[0]) {
      this.decorations = playgroundLinkMatcher.updateDeco(update, this.decorations);
    }
  },
  { decorations: instance => instance.decorations },
);

const markdownFileNavigation = (openTarget: (target: string) => boolean) =>
  EditorView.domEventHandlers({
    mousedown(event, view) {
      if (event.button !== 0) {
        return false;
      }
      const position = view.posAtCoords({ x: event.clientX, y: event.clientY });
      if (position === null) {
        return false;
      }
      const line = view.state.doc.lineAt(position);
      const offset = position - line.from;
      const matches = line.text.matchAll(/\[([^\]\n]+)\]\(([a-z][a-z0-9+.-]*:\/\/[^)\s]+)\)/gi);
      for (const match of matches) {
        const start = match.index;
        if (offset >= start && offset <= start + match[0].length && openTarget(match[2])) {
          event.preventDefault();
          return true;
        }
      }
      return false;
    },
  });

const connectTransport = (url: string): Promise<Transport> =>
  new Promise((resolve, reject) => {
    const socket = new WebSocket(url);
    const handlers = new Set<(message: string) => void>();

    socket.addEventListener('open', () => {
      resolve({
        send(message) {
          socket.send(message);
        },
        subscribe(handler) {
          handlers.add(handler);
        },
        unsubscribe(handler) {
          handlers.delete(handler);
        },
      });
    });
    socket.addEventListener('message', event => {
      handlers.forEach(handler => handler(String(event.data)));
    });
    socket.addEventListener('close', () => {
      lspStatus.textContent = 'rust-analyzer disconnected';
      lspStatus.dataset.state = 'error';
    });
    socket.addEventListener('error', () => reject(new Error('WebSocket connection failed')));
  });

const loadInfo = async (): Promise<PlaygroundInfo> => {
  const response = await fetch('/api/info');
  if (!response.ok) {
    throw new Error(`Unable to load playground: ${response.status}`);
  }
  return response.json() as Promise<PlaygroundInfo>;
};

const runCode = async (exampleId: string, files: StoredFiles): Promise<RunResult> => {
  const response = await fetch('/api/run', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ exampleId, files }),
  });
  return response.json() as Promise<RunResult>;
};

const loadStoredFiles = (): StoredFiles => {
  try {
    const value = localStorage.getItem(STORAGE_KEY);
    return value ? (JSON.parse(value) as StoredFiles) : {};
  } catch {
    return {};
  }
};

const saveStoredFile = (path: string, source: string) => {
  localStorage.setItem(STORAGE_KEY, JSON.stringify({ ...loadStoredFiles(), [path]: source }));
};

const main = async () => {
  const info = await loadInfo();
  const initialExample = info.examples[0];
  if (!initialExample) {
    throw new Error('No runnable examples found');
  }

  const storedFiles = loadStoredFiles();
  const allFiles = info.examples.flatMap(example => example.files.map(file => ({ example, file })));
  const fileByUri = new Map(allFiles.map(entry => [entry.file.uri, entry]));
  const views = new Map<string, EditorView>();
  const containers = new Map<string, HTMLDivElement>();
  const markdownViews = new Map<string, EditorView>();
  const markdownContainers = new Map<string, HTMLDivElement>();
  const exampleButtons = new Map<string, HTMLButtonElement>();
  const fileButtons = new Map<string, HTMLButtonElement>();
  let activeExampleId = initialExample.id;
  let activeUri = '';

  const renderFileTree = (example: PlaygroundExample) => {
    fileNav.replaceChildren();
    fileButtons.clear();

    const root = document.createElement('div');
    root.className = 'tree-folder tree-root';
    root.textContent = `${example.id}://`;
    fileNav.append(root);

    const rootFiles = example.files.filter(file => !file.path.includes('/'));
    const sourceFiles = example.files.filter(file => file.path.startsWith('src/'));

    const addFileButton = (file: PlaygroundFile, depth: 'root' | 'child') => {
      const button = document.createElement('button');
      button.className = 'file-tree-item';
      button.dataset.active = String(file.uri === activeUri);
      button.dataset.depth = depth;
      button.dataset.kind = file.languageId === 'rust' ? 'rs' : 'toml';
      button.type = 'button';
      button.textContent = depth === 'child' ? file.path.replace('src/', '') : file.path;
      button.addEventListener('click', () => showFile(file.uri));
      fileNav.append(button);
      fileButtons.set(file.uri, button);
    };

    rootFiles.forEach(file => addFileButton(file, 'root'));
    if (sourceFiles.length > 0) {
      const sourceFolder = document.createElement('div');
      sourceFolder.className = 'tree-folder tree-child-folder';
      sourceFolder.textContent = 'src';
      fileNav.append(sourceFolder);
      sourceFiles.forEach(file => addFileButton(file, 'child'));
    }
  };

  const selectExample = (exampleId: string, selectDefaultFile = true) => {
    const example = info.examples.find(candidate => candidate.id === exampleId);
    if (!example) {
      return false;
    }
    activeExampleId = example.id;
    exampleButtons.forEach((button, candidate) => {
      button.dataset.active = String(candidate === example.id);
    });
    markdownContainers.forEach((container, candidate) => {
      container.hidden = candidate !== example.id;
    });
    guidePath.textContent = `${example.id}://${example.guide.path}`;
    renderFileTree(example);
    if (selectDefaultFile) {
      const current = fileByUri.get(activeUri);
      const file = current?.example.id === example.id ? current.file : example.files.find(candidate => candidate.path === 'src/main.rs');
      if (file) {
        showFile(file.uri);
      }
    }
    return true;
  };

  const showFile = (uri: string) => {
    const entry = fileByUri.get(uri);
    const view = views.get(uri);
    if (!entry || !view) {
      return null;
    }
    if (entry.example.id !== activeExampleId) {
      selectExample(entry.example.id, false);
    }
    activeUri = uri;
    containers.forEach((container, candidate) => {
      container.hidden = candidate !== uri;
    });
    fileButtons.forEach((button, candidate) => {
      button.dataset.active = String(candidate === uri);
    });
    activePath.textContent = `${entry.example.id}://${entry.file.path}`;
    view.focus();
    return view;
  };

  const openFileTarget = (target: string) => {
    const match = /^([a-z][a-z0-9+.-]*):\/\/(.+?)(?:#L(\d+))?$/i.exec(target);
    const example = match ? info.examples.find(candidate => candidate.id === match[1]) : undefined;
    const file = example?.files.find(candidate => candidate.path === match?.[2]);
    const view = file ? showFile(file.uri) : null;
    if (!view) {
      return false;
    }
    const requestedLine = Number(match?.[3] ?? 1);
    const line = view.state.doc.line(Math.min(Math.max(requestedLine, 1), view.state.doc.lines));
    view.dispatch({
      selection: { anchor: line.from },
      effects: EditorView.scrollIntoView(line.from, { y: 'center' }),
    });
    return true;
  };

  info.examples.forEach(example => {
    const button = document.createElement('button');
    button.className = 'example-item';
    button.dataset.active = String(example.id === activeExampleId);
    button.type = 'button';
    button.textContent = example.title;
    button.addEventListener('click', () => selectExample(example.id));
    exampleNav.append(button);
    exampleButtons.set(example.id, button);
  });

  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const transport = await connectTransport(`${protocol}//${location.host}/lsp`);
  const client = new LSPClient({
    rootUri: info.rootUri,
    timeout: 15_000,
    workspace: lspClient => new PlaygroundWorkspace(lspClient, showFile),
    extensions: languageServerExtensions(),
    notificationHandlers: {
      'textDocument/publishDiagnostics': (_client, params) => {
        const diagnostics = Array.isArray(params?.diagnostics) ? params.diagnostics : [];
        const filename = String(params?.uri ?? '').split('/').at(-1) ?? 'file';
        diagnosticStatus.textContent = `${diagnostics.length} diagnostic${diagnostics.length === 1 ? '' : 's'} in ${filename}`;
        diagnosticStatus.dataset.state = diagnostics.length === 0 ? 'ready' : 'warning';
        return false;
      },
    },
  }).connect(transport);

  await client.initializing;

  allFiles.forEach(({ file }) => {
    const container = document.createElement('div');
    container.className = 'editor-container';
    container.hidden = true;
    editorsParent.append(container);
    containers.set(file.uri, container);

    const languageExtensions = file.languageId === 'rust'
      ? [rust(), client.plugin(file.uri, 'rust'), commandClickDefinition]
      : [];
    const view = new EditorView({
      doc: storedFiles[file.target] ?? file.source,
      extensions: [
        basicSetup,
        oneDark,
        languageExtensions,
        EditorView.lineWrapping,
        EditorView.updateListener.of(update => {
          if (update.docChanged) {
            saveStoredFile(file.target, update.state.doc.toString());
          }
        }),
      ],
      parent: container,
    });
    views.set(file.uri, view);
  });

  info.examples.forEach(example => {
    const container = document.createElement('div');
    container.className = 'markdown-container';
    container.hidden = example.id !== activeExampleId;
    markdownEditorParent.append(container);
    markdownContainers.set(example.id, container);

    const view = new EditorView({
      doc: storedFiles[example.guide.target] ?? example.guide.source,
      extensions: [
        basicSetup,
        markdown(),
        oneDark,
        playgroundLinkDecorations,
        markdownFileNavigation(openFileTarget),
        EditorView.lineWrapping,
        EditorView.updateListener.of(update => {
          if (update.docChanged) {
            saveStoredFile(example.guide.target, update.state.doc.toString());
          }
        }),
      ],
      parent: container,
    });
    markdownViews.set(example.id, view);
  });

  selectExample(activeExampleId);
  lspStatus.textContent = 'rust-analyzer connected';
  lspStatus.dataset.state = 'ready';
  if (diagnosticStatus.textContent === 'Diagnostics pending') {
    const rustFileCount = allFiles.filter(entry => entry.file.languageId === 'rust').length;
    diagnosticStatus.textContent = `${rustFileCount} Rust files open in LSP`;
    diagnosticStatus.dataset.state = 'ready';
  }
  revision.textContent = info.revision.slice(0, 8);

  runButton.addEventListener('click', async () => {
    const example = info.examples.find(candidate => candidate.id === activeExampleId);
    if (!example) {
      return;
    }
    runButton.disabled = true;
    runStatus.textContent = `Running ${example.title}`;
    output.textContent = '';
    const files = Object.fromEntries(
      example.files.map(file => [file.path, views.get(file.uri)?.state.doc.toString() ?? file.source]),
    );
    const result = await runCode(example.id, files);
    output.textContent = result.output;
    runStatus.textContent = result.success ? 'Finished' : 'Failed';
    runStatus.dataset.state = result.success ? 'ready' : 'error';
    runButton.disabled = false;
  });

  resetButton.addEventListener('click', () => {
    const example = info.examples.find(candidate => candidate.id === activeExampleId);
    if (!example) {
      return;
    }
    example.files.forEach(file => {
      const view = views.get(file.uri);
      view?.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: file.source } });
    });
    const markdownView = markdownViews.get(example.id);
    markdownView?.dispatch({
      changes: { from: 0, to: markdownView.state.doc.length, insert: example.guide.source },
    });
  });
};

main().catch(error => {
  lspStatus.textContent = error instanceof Error ? error.message : 'Unable to start';
  lspStatus.dataset.state = 'error';
});
