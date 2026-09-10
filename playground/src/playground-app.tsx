import { useMutation, useQuery } from '@tanstack/react-query';
import { useCallback, useMemo, useRef, useState } from 'react';

import { loadInfo, persistFile, runCode } from './api';
import { CodeEditors, type CodeEditorsHandle } from './code-editors';
import { FileTreePanel } from './file-tree-panel';
import { MarkdownEditor, type MarkdownEditorHandle } from './markdown-editor';
import { createFileSaver } from './persistence';
import type { PlaygroundExample, PlaygroundInfo } from './types';

type ConnectionState = 'connecting' | 'ready' | 'error';

const LoadingScreen = () => (
  <main className="load-screen" aria-busy="true" aria-live="polite">
    <div className="spinner" />
    <p className="eyebrow">LOADING WORKSPACE</p>
    <h1>Reading examples and starting the editor</h1>
  </main>
);

const ErrorScreen = ({ message, retry }: { message: string; retry: () => void }) => (
  <main className="load-screen" role="alert">
    <p className="eyebrow">WORKSPACE ERROR</p>
    <h1>{message}</h1>
    <button className="button primary" type="button" onClick={retry}>Try again</button>
  </main>
);

const initialFile = (example: PlaygroundExample) =>
  example.files.find(file => file.path === 'src/main.rs') ?? example.files[0];

const Playground = ({ info }: { info: PlaygroundInfo }) => {
  const firstExample = info.examples[0];
  const [activeExampleId, setActiveExampleId] = useState(firstExample?.id ?? '');
  const [activeUri, setActiveUri] = useState(firstExample ? initialFile(firstExample)?.uri ?? '' : '');
  const [connection, setConnection] = useState<ConnectionState>('connecting');
  const [connectionError, setConnectionError] = useState('');
  const [diagnostics, setDiagnostics] = useState('Diagnostics pending');
  const [diagnosticState, setDiagnosticState] = useState<'ready' | 'warning'>('ready');
  const [saveError, setSaveError] = useState('');
  const codeEditors = useRef<CodeEditorsHandle>(null);
  const markdownEditor = useRef<MarkdownEditorHandle>(null);

  const saveMutation = useMutation({
    mutationFn: ({ target, source }: { target: string; source: string }) => persistFile(target, source),
    onError: (_error, variables) => setSaveError(`Save failed: ${variables.target}`),
    onSuccess: () => setSaveError(''),
  });
  const saveFile = useMemo(
    () => createFileSaver((target, source) => saveMutation.mutateAsync({ target, source })),
    [saveMutation.mutateAsync],
  );

  const activeExample = info.examples.find(example => example.id === activeExampleId);
  const activeFile = activeExample?.files.find(file => file.uri === activeUri);

  const openFile = useCallback((uri: string) => {
    const entry = info.examples
      .flatMap(example => example.files.map(file => ({ example, file })))
      .find(candidate => candidate.file.uri === uri);
    if (!entry) {
      return false;
    }
    setActiveExampleId(entry.example.id);
    setActiveUri(uri);
    return true;
  }, [info.examples]);

  const openFileTarget = useCallback((target: string) => {
    const match = /^([a-z][a-z0-9+.-]*):\/\/(.+?)(?:#L(\d+))?$/i.exec(target);
    const example = match ? info.examples.find(candidate => candidate.id === match[1]) : undefined;
    const file = example?.files.find(candidate => candidate.path === match?.[2]);
    if (!file || !openFile(file.uri)) {
      return false;
    }
    codeEditors.current?.goToLine(file.uri, Number(match?.[3] ?? 1));
    return true;
  }, [info.examples, openFile]);

  const selectExample = (example: PlaygroundExample) => {
    const file = initialFile(example);
    setActiveExampleId(example.id);
    if (file) {
      setActiveUri(file.uri);
    }
  };

  const runMutation = useMutation({
    mutationFn: async () => {
      if (!activeExample) {
        throw new Error('No runnable example selected');
      }
      return runCode(activeExample.id, codeEditors.current?.getFiles(activeExample) ?? {});
    },
  });

  const reset = () => {
    if (activeExample) {
      codeEditors.current?.reset(activeExample);
    }
    markdownEditor.current?.reset();
  };

  if (!firstExample) {
    return <ErrorScreen message="No runnable examples found" retry={() => window.location.reload()} />;
  }

  const output = runMutation.data?.output
    || (runMutation.error instanceof Error ? runMutation.error.message : '')
    || 'Press Run to compile against the cloned LiteLLM repository.';
  const runStatus = runMutation.isPending
    ? `Running ${activeExample?.title ?? 'example'}`
    : runMutation.data ? (runMutation.data.success ? 'Finished' : 'Failed') : '';
  const rustFileCount = info.examples.flatMap(example => example.files).filter(file => file.languageId === 'rust').length;

  return (
    <main className="shell">
      <header className="topbar">
        <div>
          <span className="eyebrow">PERSISTENT PLAYGROUND</span>
          <h1>LiteLLM Rust Playground</h1>
        </div>
        <div className="status-row">
          <span className="status" data-state={connection === 'connecting' ? 'loading' : connection}>
            {connection === 'ready' ? 'rust-analyzer connected' : connectionError || 'Connecting to rust-analyzer'}
          </span>
          <span className="status" data-state={diagnosticState}>{diagnostics}</span>
          {(saveMutation.isPending || saveError) && (
            <span className="status" data-state={saveError ? 'error' : 'loading'}>
              {saveError || 'Saving'}
            </span>
          )}
          <button className="button secondary" type="button" onClick={reset}>Reset</button>
          <button
            className="button primary"
            type="button"
            disabled={runMutation.isPending || connection !== 'ready'}
            onClick={() => runMutation.mutate()}
          >
            {runMutation.isPending ? <><span className="button-spinner" />Running</> : 'Run'}
          </button>
        </div>
      </header>

      <section className="workspace" aria-busy={connection === 'connecting'}>
        <aside className="docs" aria-label="Markdown document">
          <div className="docbar">
            <span>{info.guide.path}</span>
            <span className="document-language">Markdown</span>
          </div>
          <MarkdownEditor ref={markdownEditor} guide={info.guide} onOpenTarget={openFileTarget} onSave={saveFile} />
        </aside>

        <section className="editor-panel" aria-label="Rust editor">
          <div className="filebar">
            <span>{activeExample && activeFile ? `${activeExample.id}://${activeFile.path}` : 'Select a file'}</span>
            <span className="revision">{info.revision.slice(0, 8)}</span>
          </div>
          <div className="editor-body">
            <div className="editor-stack">
              <CodeEditors
                ref={codeEditors}
                info={info}
                activeUri={activeUri}
                onOpenFile={openFile}
                onSave={saveFile}
                onConnectionChange={(state, message) => {
                  setConnection(state);
                  setConnectionError(message ?? '');
                  if (state === 'ready' && diagnostics === 'Diagnostics pending') {
                    setDiagnostics(`${rustFileCount} Rust files open in LSP`);
                  }
                }}
                onDiagnostics={(message, hasDiagnostics) => {
                  setDiagnostics(message);
                  setDiagnosticState(hasDiagnostics ? 'warning' : 'ready');
                }}
              />
              {connection === 'connecting' && (
                <div className="editor-loading" role="status">
                  <div className="spinner" />
                  <span>Starting rust-analyzer</span>
                </div>
              )}
            </div>
            <aside className="file-tree" aria-label="Project files">
              <p className="tree-title">EXAMPLES</p>
              <nav className="example-nav" aria-label="Examples">
                {info.examples.map(example => (
                  <button
                    key={example.id}
                    className="example-item"
                    data-active={example.id === activeExampleId}
                    type="button"
                    onClick={() => selectExample(example)}
                  >
                    {example.title}
                  </button>
                ))}
              </nav>
              {activeExample && <FileTreePanel example={activeExample} activePath={activeFile?.path} onOpenFile={openFile} />}
            </aside>
          </div>
        </section>

        <section className="output-panel" aria-live="polite">
          <div className="output-header">
            <span>OUTPUT</span>
            <span data-state={runMutation.data?.success === false || runMutation.isError ? 'error' : 'ready'}>{runStatus}</span>
          </div>
          <pre>{output || 'Program finished without output'}</pre>
        </section>
      </section>
    </main>
  );
};

export const PlaygroundApp = () => {
  const infoQuery = useQuery({ queryKey: ['playground-info'], queryFn: loadInfo });

  if (infoQuery.isPending) {
    return <LoadingScreen />;
  }
  if (infoQuery.isError) {
    const message = infoQuery.error instanceof Error ? infoQuery.error.message : 'Unable to load workspace';
    return <ErrorScreen message={message} retry={() => { void infoQuery.refetch(); }} />;
  }
  return <Playground info={infoQuery.data} />;
};
