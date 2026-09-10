import {
  jumpToDefinition,
  LSPClient,
  LSPPlugin,
  Workspace,
  type Transport,
  type WorkspaceFile,
} from '@codemirror/lsp-client';
import { EditorView } from '@codemirror/view';

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

export class PlaygroundWorkspace extends Workspace {
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

export const commandClickDefinition = EditorView.domEventHandlers({
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

export const connectTransport = (
  url: string,
  onClose: () => void,
): Promise<Transport> => new Promise((resolve, reject) => {
  const socket = new WebSocket(url);
  const handlers = new Set<(message: string) => void>();

  socket.addEventListener('open', () => {
    resolve({
      send: message => socket.send(message),
      subscribe: handler => handlers.add(handler),
      unsubscribe: handler => handlers.delete(handler),
    });
  });
  socket.addEventListener('message', event => {
    handlers.forEach(handler => handler(String(event.data)));
  });
  socket.addEventListener('close', onClose);
  socket.addEventListener('error', () => reject(new Error('WebSocket connection failed')));
});
