import { rust } from '@codemirror/lang-rust';
import { languageServerExtensions, LSPClient } from '@codemirror/lsp-client';
import { EditorView, basicSetup } from 'codemirror';
import { forwardRef, useEffect, useImperativeHandle, useRef } from 'react';

import { commandClickDefinition, connectTransport, PlaygroundWorkspace } from './lsp';
import { playgroundEditorTheme } from './editor-theme';
import type { PlaygroundExample, PlaygroundInfo, StoredFiles } from './types';

export type CodeEditorsHandle = {
  getFiles: (example: PlaygroundExample) => StoredFiles;
  goToLine: (uri: string, requestedLine: number) => void;
  reset: (example: PlaygroundExample) => void;
};

type CodeEditorsProps = {
  activeUri: string;
  info: PlaygroundInfo;
  onConnectionChange: (state: 'connecting' | 'ready' | 'error', message?: string) => void;
  onDiagnostics: (message: string, hasDiagnostics: boolean) => void;
  onOpenFile: (uri: string) => boolean;
  onSave: (target: string, source: string) => void;
};

export const CodeEditors = forwardRef<CodeEditorsHandle, CodeEditorsProps>(function CodeEditors(
  { activeUri, info, onConnectionChange, onDiagnostics, onOpenFile, onSave },
  handle,
) {
  const containers = useRef(new Map<string, HTMLDivElement>());
  const views = useRef(new Map<string, EditorView>());
  const callbacks = useRef({ onConnectionChange, onDiagnostics, onOpenFile, onSave });
  callbacks.current = { onConnectionChange, onDiagnostics, onOpenFile, onSave };

  useImperativeHandle(handle, () => ({
    getFiles: example => Object.fromEntries(
      example.files.map(file => [file.path, views.current.get(file.uri)?.state.doc.toString() ?? file.source]),
    ),
    goToLine: (uri, requestedLine) => {
      const view = views.current.get(uri);
      if (!view) {
        return;
      }
      const line = view.state.doc.line(Math.min(Math.max(requestedLine, 1), view.state.doc.lines));
      view.dispatch({
        selection: { anchor: line.from },
        effects: EditorView.scrollIntoView(line.from, { y: 'center' }),
      });
      view.focus();
    },
    reset: example => example.files.forEach(file => {
      const view = views.current.get(file.uri);
      view?.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: file.source } });
    }),
  }), []);

  useEffect(() => {
    let disposed = false;
    const setup = async () => {
      callbacks.current.onConnectionChange('connecting');
      const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
      const transport = await connectTransport(`${protocol}//${location.host}/lsp`, () => {
        if (!disposed) {
          callbacks.current.onConnectionChange('error', 'rust-analyzer disconnected');
        }
      });
      const client = new LSPClient({
        rootUri: info.rootUri,
        timeout: 15_000,
        workspace: lspClient => new PlaygroundWorkspace(lspClient, uri => {
          callbacks.current.onOpenFile(uri);
          return views.current.get(uri) ?? null;
        }),
        extensions: languageServerExtensions(),
        notificationHandlers: {
          'textDocument/publishDiagnostics': (_client, params) => {
            const diagnostics = Array.isArray(params?.diagnostics) ? params.diagnostics : [];
            const filename = String(params?.uri ?? '').split('/').at(-1) ?? 'file';
            callbacks.current.onDiagnostics(
              `${diagnostics.length} diagnostic${diagnostics.length === 1 ? '' : 's'} in ${filename}`,
              diagnostics.length > 0,
            );
            return false;
          },
        },
      }).connect(transport);
      await client.initializing;
      if (disposed) {
        return;
      }
      info.examples.flatMap(example => example.files).forEach(file => {
        const parent = containers.current.get(file.uri);
        if (!parent) {
          return;
        }
        const languageExtensions = file.languageId === 'rust'
          ? [rust(), client.plugin(file.uri, 'rust'), commandClickDefinition]
          : [];
        const view = new EditorView({
          doc: file.source,
          extensions: [
            basicSetup,
            playgroundEditorTheme,
            languageExtensions,
            EditorView.lineWrapping,
            EditorView.updateListener.of(update => {
              if (update.docChanged) {
                callbacks.current.onSave(file.target, update.state.doc.toString());
              }
            }),
          ],
          parent,
        });
        views.current.set(file.uri, view);
      });
      callbacks.current.onConnectionChange('ready');
    };

    setup().catch(error => {
      const message = error instanceof Error ? error.message : 'Unable to start rust-analyzer';
      callbacks.current.onConnectionChange('error', message);
    });
    return () => {
      disposed = true;
      views.current.forEach(view => view.destroy());
      views.current.clear();
    };
  }, [info]);

  return (
    <div className="editors">
      {info.examples.flatMap(example => example.files).map(file => (
        <div
          key={file.uri}
          className="editor-container"
          hidden={file.uri !== activeUri}
          ref={element => {
            if (element) {
              containers.current.set(file.uri, element);
            } else {
              containers.current.delete(file.uri);
            }
          }}
        />
      ))}
    </div>
  );
});
