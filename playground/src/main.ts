import { rust } from '@codemirror/lang-rust';
import { markdown } from '@codemirror/lang-markdown';
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
import { EditorView, basicSetup } from 'codemirror';

import './style.css';

const STORAGE_KEY = 'litellm-rust-playground-files';
const GUIDE_PATH = 'GUIDE.md';
const DEFAULT_GUIDE = `# Prove the connection

Cmd-click \`build_proof\` in \`src/main.rs\`. The Rust editor opens its definition in \`src/proof.rs\`.

Hover \`litellm_core::Error\` for dependency type information.

## What this uses

- **Editor:** CodeMirror 6
- **Language server:** rust-analyzer over WebSocket
- **Dependency:** in-repo \`litellm-core\` path dependency
- **Proc macro:** Serde \`Serialize\` and \`Deserialize\` derives

> This Markdown and the Rust source are saved in this browser. Reset restores the example.
`;

type PlaygroundFile = {
  path: string;
  uri: string;
  source: string;
};

type PlaygroundInfo = {
  files: PlaygroundFile[];
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
const fileNav = requireElement<HTMLElement>('file-nav');
const activePath = requireElement<HTMLSpanElement>('active-path');

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

const runCode = async (files: StoredFiles): Promise<RunResult> => {
  const response = await fetch('/api/run', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ files }),
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
  const storedFiles = loadStoredFiles();
  const views = new Map<string, EditorView>();
  const containers = new Map<string, HTMLDivElement>();
  const buttons = new Map<string, HTMLButtonElement>();
  let activeUri = info.files[0]?.uri ?? '';

  const markdownView = new EditorView({
    doc: storedFiles[GUIDE_PATH] ?? DEFAULT_GUIDE,
    extensions: [
      basicSetup,
      markdown(),
      oneDark,
      EditorView.lineWrapping,
      EditorView.updateListener.of(update => {
        if (update.docChanged) {
          saveStoredFile(GUIDE_PATH, update.state.doc.toString());
        }
      }),
    ],
    parent: markdownEditorParent,
  });

  const showFile = (uri: string) => {
    const view = views.get(uri);
    if (!view) {
      return null;
    }
    activeUri = uri;
    containers.forEach((container, candidate) => {
      container.hidden = candidate !== uri;
    });
    buttons.forEach((button, candidate) => {
      button.dataset.active = String(candidate === uri);
    });
    activePath.textContent = info.files.find(file => file.uri === uri)?.path ?? uri;
    view.focus();
    return view;
  };

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

  info.files.forEach(file => {
    const button = document.createElement('button');
    button.className = 'file-tree-item';
    button.type = 'button';
    button.textContent = file.path.replace('src/', '');
    button.addEventListener('click', () => showFile(file.uri));
    fileNav.append(button);
    buttons.set(file.uri, button);

    const container = document.createElement('div');
    container.className = 'editor-container';
    container.hidden = file.uri !== activeUri;
    editorsParent.append(container);
    containers.set(file.uri, container);

    const view = new EditorView({
      doc: storedFiles[file.path] ?? file.source,
      extensions: [
        basicSetup,
        rust(),
        oneDark,
        client.plugin(file.uri, 'rust'),
        commandClickDefinition,
        EditorView.lineWrapping,
        EditorView.updateListener.of(update => {
          if (!update.docChanged) {
            return;
          }
          saveStoredFile(file.path, update.state.doc.toString());
        }),
      ],
      parent: container,
    });
    views.set(file.uri, view);
  });

  showFile(activeUri);
  lspStatus.textContent = 'rust-analyzer connected';
  lspStatus.dataset.state = 'ready';
  if (diagnosticStatus.textContent === 'Diagnostics pending') {
    diagnosticStatus.textContent = `${info.files.length} files open in LSP`;
    diagnosticStatus.dataset.state = 'ready';
  }
  revision.textContent = info.revision.slice(0, 8);

  runButton.addEventListener('click', async () => {
    runButton.disabled = true;
    runStatus.textContent = 'Compiling';
    output.textContent = '';
    const files = Object.fromEntries(
      info.files.map(file => [file.path, views.get(file.uri)?.state.doc.toString() ?? file.source]),
    );
    const result = await runCode(files);
    output.textContent = result.output;
    runStatus.textContent = result.success ? 'Finished' : 'Failed';
    runStatus.dataset.state = result.success ? 'ready' : 'error';
    runButton.disabled = false;
  });

  resetButton.addEventListener('click', () => {
    localStorage.removeItem(STORAGE_KEY);
    info.files.forEach(file => {
      const view = views.get(file.uri);
      view?.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: file.source } });
    });
    markdownView.dispatch({
      changes: { from: 0, to: markdownView.state.doc.length, insert: DEFAULT_GUIDE },
    });
  });
};

main().catch(error => {
  lspStatus.textContent = error instanceof Error ? error.message : 'Unable to start';
  lspStatus.dataset.state = 'error';
});
