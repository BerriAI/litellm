import { markdown } from '@codemirror/lang-markdown';
import { rust } from '@codemirror/lang-rust';
import { languageServerExtensions, LSPClient } from '@codemirror/lsp-client';
import { oneDark } from '@codemirror/theme-one-dark';
import { EditorView, basicSetup } from 'codemirror';

import { loadInfo, runCode } from './api';
import { elements } from './dom';
import { ExampleFileTree } from './file-tree';
import { commandClickDefinition, connectTransport, PlaygroundWorkspace } from './lsp';
import { markdownFileNavigation, playgroundLinkDecorations } from './markdown-navigation';
import { createFileSaver } from './persistence';
import type { PlaygroundExample, PlaygroundInfo } from './types';

class PlaygroundApp {
  private activeExampleId = '';
  private activeUri = '';
  private readonly containers = new Map<string, HTMLDivElement>();
  private readonly exampleButtons = new Map<string, HTMLButtonElement>();
  private readonly views = new Map<string, EditorView>();
  private info!: PlaygroundInfo;
  private markdownView!: EditorView;
  private readonly saveFile = createFileSaver(target => {
    elements.runStatus.textContent = `Save failed: ${target}`;
    elements.runStatus.dataset.state = 'error';
  });
  private readonly fileTree = new ExampleFileTree(
    elements.fileNav,
    elements.fileTreeTitle,
    uri => { this.showFile(uri); },
  );

  public async start() {
    this.info = await loadInfo();
    const initialExample = this.info.examples[0];
    if (!initialExample) {
      throw new Error('No runnable examples found');
    }
    this.activeExampleId = initialExample.id;

    this.renderExampleNavigation();
    const client = await this.connectLsp();
    this.createCodeEditors(client);
    this.createMarkdownEditor();
    this.finishSetup();
  }

  private get allFiles() {
    return this.info.examples.flatMap(example => example.files.map(file => ({ example, file })));
  }

  private get fileByUri() {
    return new Map(this.allFiles.map(entry => [entry.file.uri, entry]));
  }

  private renderExampleNavigation() {
    this.info.examples.forEach(example => {
      const button = document.createElement('button');
      button.className = 'example-item';
      button.dataset.active = String(example.id === this.activeExampleId);
      button.type = 'button';
      button.textContent = example.title;
      button.addEventListener('click', () => this.selectExample(example.id));
      elements.exampleNav.append(button);
      this.exampleButtons.set(example.id, button);
    });
  }

  private async connectLsp() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const transport = await connectTransport(`${protocol}//${location.host}/lsp`, () => {
      elements.lspStatus.textContent = 'rust-analyzer disconnected';
      elements.lspStatus.dataset.state = 'error';
    });
    const client = new LSPClient({
      rootUri: this.info.rootUri,
      timeout: 15_000,
      workspace: lspClient => new PlaygroundWorkspace(lspClient, uri => this.showFile(uri)),
      extensions: languageServerExtensions(),
      notificationHandlers: {
        'textDocument/publishDiagnostics': (_client, params) => {
          const diagnostics = Array.isArray(params?.diagnostics) ? params.diagnostics : [];
          const filename = String(params?.uri ?? '').split('/').at(-1) ?? 'file';
          elements.diagnosticStatus.textContent = `${diagnostics.length} diagnostic${diagnostics.length === 1 ? '' : 's'} in ${filename}`;
          elements.diagnosticStatus.dataset.state = diagnostics.length === 0 ? 'ready' : 'warning';
          return false;
        },
      },
    }).connect(transport);
    await client.initializing;
    return client;
  }

  private createCodeEditors(client: LSPClient) {
    this.allFiles.forEach(({ file }) => {
      const container = document.createElement('div');
      container.className = 'editor-container';
      container.hidden = true;
      elements.editorsParent.append(container);
      this.containers.set(file.uri, container);

      const languageExtensions = file.languageId === 'rust'
        ? [rust(), client.plugin(file.uri, 'rust'), commandClickDefinition]
        : [];
      const view = new EditorView({
        doc: file.source,
        extensions: [
          basicSetup,
          oneDark,
          languageExtensions,
          EditorView.lineWrapping,
          EditorView.updateListener.of(update => {
            if (update.docChanged) {
              this.saveFile(file.target, update.state.doc.toString());
            }
          }),
        ],
        parent: container,
      });
      this.views.set(file.uri, view);
    });
  }

  private createMarkdownEditor() {
    this.markdownView = new EditorView({
      doc: this.info.guide.source,
      extensions: [
        basicSetup,
        markdown(),
        oneDark,
        playgroundLinkDecorations,
        markdownFileNavigation(target => this.openFileTarget(target)),
        EditorView.lineWrapping,
        EditorView.updateListener.of(update => {
          if (update.docChanged) {
            this.saveFile(this.info.guide.target, update.state.doc.toString());
          }
        }),
      ],
      parent: elements.markdownEditorParent,
    });
  }

  private selectExample(exampleId: string, selectDefaultFile = true) {
    const example = this.info.examples.find(candidate => candidate.id === exampleId);
    if (!example) {
      return false;
    }
    this.activeExampleId = example.id;
    this.exampleButtons.forEach((button, candidate) => {
      button.dataset.active = String(candidate === example.id);
    });
    const current = this.fileByUri.get(this.activeUri);
    const currentPath = current?.example.id === example.id ? current.file.path : undefined;
    this.fileTree.render(example, currentPath);
    if (selectDefaultFile) {
      const file = currentPath
        ? current?.file
        : example.files.find(candidate => candidate.path === 'src/main.rs');
      if (file) {
        this.showFile(file.uri);
      }
    }
    return true;
  }

  private showFile(uri: string) {
    const entry = this.fileByUri.get(uri);
    const view = this.views.get(uri);
    if (!entry || !view) {
      return null;
    }
    if (entry.example.id !== this.activeExampleId) {
      this.selectExample(entry.example.id, false);
    }
    this.activeUri = uri;
    this.containers.forEach((container, candidate) => {
      container.hidden = candidate !== uri;
    });
    this.fileTree.select(entry.file.path);
    elements.activePath.textContent = `${entry.example.id}://${entry.file.path}`;
    view.focus();
    return view;
  }

  private openFileTarget(target: string) {
    const match = /^([a-z][a-z0-9+.-]*):\/\/(.+?)(?:#L(\d+))?$/i.exec(target);
    const example = match ? this.info.examples.find(candidate => candidate.id === match[1]) : undefined;
    const file = example?.files.find(candidate => candidate.path === match?.[2]);
    const view = file ? this.showFile(file.uri) : null;
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
  }

  private finishSetup() {
    elements.guidePath.textContent = this.info.guide.path;
    this.selectExample(this.activeExampleId);
    elements.lspStatus.textContent = 'rust-analyzer connected';
    elements.lspStatus.dataset.state = 'ready';
    if (elements.diagnosticStatus.textContent === 'Diagnostics pending') {
      const rustFileCount = this.allFiles.filter(entry => entry.file.languageId === 'rust').length;
      elements.diagnosticStatus.textContent = `${rustFileCount} Rust files open in LSP`;
      elements.diagnosticStatus.dataset.state = 'ready';
    }
    elements.revision.textContent = this.info.revision.slice(0, 8);
    elements.runButton.addEventListener('click', () => { void this.runActiveExample(); });
    elements.resetButton.addEventListener('click', () => this.reset());
  }

  private async runActiveExample() {
    const example = this.activeExample;
    if (!example) {
      return;
    }
    elements.runButton.disabled = true;
    elements.runStatus.textContent = `Running ${example.title}`;
    elements.output.textContent = '';
    try {
      const files = Object.fromEntries(
        example.files.map(file => [file.path, this.views.get(file.uri)?.state.doc.toString() ?? file.source]),
      );
      const result = await runCode(example.id, files);
      elements.output.textContent = result.output;
      elements.runStatus.textContent = result.success ? 'Finished' : 'Failed';
      elements.runStatus.dataset.state = result.success ? 'ready' : 'error';
    } catch (error) {
      elements.runStatus.textContent = error instanceof Error ? error.message : 'Run failed';
      elements.runStatus.dataset.state = 'error';
    } finally {
      elements.runButton.disabled = false;
    }
  }

  private get activeExample(): PlaygroundExample | undefined {
    return this.info.examples.find(candidate => candidate.id === this.activeExampleId);
  }

  private reset() {
    this.activeExample?.files.forEach(file => {
      const view = this.views.get(file.uri);
      view?.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: file.source } });
    });
    this.markdownView.dispatch({
      changes: { from: 0, to: this.markdownView.state.doc.length, insert: this.info.guide.source },
    });
  }
}

export const startPlayground = () => {
  new PlaygroundApp().start().catch(error => {
    elements.lspStatus.textContent = error instanceof Error ? error.message : 'Unable to start';
    elements.lspStatus.dataset.state = 'error';
  });
};
