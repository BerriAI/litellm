import { HighlightStyle, syntaxHighlighting } from '@codemirror/language';
import { EditorView } from '@codemirror/view';
import { tags } from '@lezer/highlight';

const editorTheme = EditorView.theme({
  '&': {
    color: 'var(--text-primary)',
    backgroundColor: 'var(--surface-editor)',
  },
  '.cm-content': {
    padding: '0.5rem 0',
    caretColor: 'var(--accent)',
  },
  '.cm-cursor, .cm-dropCursor': {
    borderLeftColor: 'var(--accent)',
  },
  '&.cm-focused .cm-selectionBackground, .cm-selectionBackground, ::selection': {
    backgroundColor: 'var(--selection)',
  },
  '.cm-activeLine': {
    backgroundColor: 'var(--surface-active)',
  },
  '.cm-gutters': {
    color: 'var(--text-subtle)',
    backgroundColor: 'var(--surface-editor)',
    borderRight: '1px solid var(--border-subtle)',
  },
  '.cm-activeLineGutter': {
    color: 'var(--text-secondary)',
    backgroundColor: 'var(--surface-active)',
  },
  '.cm-foldPlaceholder': {
    color: 'var(--text-secondary)',
    backgroundColor: 'var(--surface-raised)',
    borderColor: 'var(--border-strong)',
  },
  '.cm-panels, .cm-tooltip': {
    color: 'var(--text-primary)',
    backgroundColor: 'var(--surface-raised)',
    borderColor: 'var(--border-strong)',
  },
  '.cm-tooltip-autocomplete > ul > li[aria-selected]': {
    color: 'var(--text-primary)',
    backgroundColor: 'var(--surface-selected)',
  },
  '.cm-searchMatch': {
    backgroundColor: 'var(--search-match)',
    outline: '1px solid var(--warning)',
  },
  '.cm-matchingBracket': {
    color: 'var(--text-primary)',
    backgroundColor: 'var(--surface-selected)',
    outline: '1px solid var(--accent)',
  },
}, { dark: true });

const highlightStyle = HighlightStyle.define([
  { tag: [tags.keyword, tags.modifier, tags.operatorKeyword], color: 'var(--syntax-keyword)' },
  { tag: [tags.name, tags.variableName, tags.propertyName], color: 'var(--syntax-name)' },
  { tag: [tags.typeName, tags.className, tags.namespace], color: 'var(--syntax-type)' },
  { tag: [tags.function(tags.variableName), tags.labelName], color: 'var(--syntax-function)' },
  { tag: [tags.string, tags.special(tags.string)], color: 'var(--syntax-string)' },
  { tag: [tags.number, tags.bool, tags.null], color: 'var(--syntax-number)' },
  { tag: [tags.comment, tags.meta], color: 'var(--syntax-comment)', fontStyle: 'italic' },
  { tag: [tags.heading, tags.strong], color: 'var(--syntax-heading)', fontWeight: '700' },
  { tag: [tags.link, tags.url], color: 'var(--syntax-link)', textDecoration: 'underline' },
  { tag: [tags.punctuation, tags.bracket], color: 'var(--text-secondary)' },
  { tag: tags.invalid, color: 'var(--syntax-invalid)', textDecoration: 'underline wavy' },
]);

export const playgroundEditorTheme = [editorTheme, syntaxHighlighting(highlightStyle)];
