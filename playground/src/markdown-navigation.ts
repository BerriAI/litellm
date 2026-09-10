import { Decoration, EditorView, MatchDecorator, ViewPlugin } from '@codemirror/view';

const linkMatcher = new MatchDecorator({
  regexp: /\[[^\]\n]+\]\([a-z][a-z0-9+.-]*:\/\/[^)\s]+\)/gi,
  decoration: Decoration.mark({ class: 'cm-playground-link' }),
});

export const playgroundLinkDecorations = ViewPlugin.fromClass(
  class {
    public decorations;

    public constructor(view: EditorView) {
      this.decorations = linkMatcher.createDeco(view);
    }

    public update(update: Parameters<typeof linkMatcher.updateDeco>[0]) {
      this.decorations = linkMatcher.updateDeco(update, this.decorations);
    }
  },
  { decorations: instance => instance.decorations },
);

export const markdownFileNavigation = (openTarget: (target: string) => boolean) =>
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
