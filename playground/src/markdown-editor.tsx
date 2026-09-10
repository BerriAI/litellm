import { markdown } from '@codemirror/lang-markdown';
import { oneDark } from '@codemirror/theme-one-dark';
import { EditorView, basicSetup } from 'codemirror';
import { forwardRef, useEffect, useImperativeHandle, useRef } from 'react';

import { markdownFileNavigation, playgroundLinkDecorations } from './markdown-navigation';
import type { PlaygroundGuide } from './types';

export type MarkdownEditorHandle = { reset: () => void };

type MarkdownEditorProps = {
  guide: PlaygroundGuide;
  onOpenTarget: (target: string) => boolean;
  onSave: (target: string, source: string) => void;
};

export const MarkdownEditor = forwardRef<MarkdownEditorHandle, MarkdownEditorProps>(function MarkdownEditor(
  { guide, onOpenTarget, onSave },
  handle,
) {
  const container = useRef<HTMLDivElement>(null);
  const view = useRef<EditorView | null>(null);
  const callbacks = useRef({ onOpenTarget, onSave });
  callbacks.current = { onOpenTarget, onSave };

  useImperativeHandle(handle, () => ({
    reset: () => view.current?.dispatch({
      changes: { from: 0, to: view.current.state.doc.length, insert: guide.source },
    }),
  }), [guide.source]);

  useEffect(() => {
    if (!container.current) {
      return;
    }
    const editor = new EditorView({
      doc: guide.source,
      extensions: [
        basicSetup,
        markdown(),
        oneDark,
        playgroundLinkDecorations,
        markdownFileNavigation(target => callbacks.current.onOpenTarget(target)),
        EditorView.lineWrapping,
        EditorView.updateListener.of(update => {
          if (update.docChanged) {
            callbacks.current.onSave(guide.target, update.state.doc.toString());
          }
        }),
      ],
      parent: container.current,
    });
    view.current = editor;
    return () => {
      editor.destroy();
      view.current = null;
    };
  }, [guide]);

  return <div className="markdown-editor" ref={container} />;
});
