export type RootBlockKey = "model" | "state" | "questions";

export interface RootBlock {
  readonly key: RootBlockKey;
  readonly startLine: number;
  readonly endLine: number;
}

export const ROOT_BLOCK_STYLES: Readonly<Record<RootBlockKey, { band: string; accent: string }>> = {
  model: { band: "bg-blue-500/10 before:bg-blue-500/60", accent: "border-l-blue-500/60" },
  state: { band: "bg-green-500/10 before:bg-green-500/60", accent: "border-l-green-500/60" },
  questions: { band: "bg-orange-500/10 before:bg-orange-500/60", accent: "border-l-orange-500/60" },
};

interface Token {
  readonly text: string;
  readonly line: number;
}

interface ScanState {
  readonly depth: number;
  readonly previous?: Token;
  readonly open?: { readonly key: RootBlockKey; readonly startLine: number };
  readonly blocks: readonly RootBlock[];
}

const JSON_TOKEN = /"(?:[^"\\\n]|\\.)*"?|[{}[\],:]|[^\s{}[\],:"]+/g;

const isRootBlockKey = (key: string): key is RootBlockKey => Object.hasOwn(ROOT_BLOCK_STYLES, key);

const tokenize = (text: string): Token[] =>
  text
    .split("\n")
    .flatMap((lineText, line) => Array.from(lineText.matchAll(JSON_TOKEN), ([match]) => ({ text: match, line })));

const closeOpenBlock = ({ open, previous, blocks }: ScanState): readonly RootBlock[] =>
  open && previous ? [...blocks, { ...open, endLine: previous.line }] : blocks;

function step(state: ScanState, token: Token): ScanState {
  const next = { ...state, previous: token };
  switch (token.text) {
    case "{":
    case "[":
      return { ...next, depth: state.depth + 1 };
    case "}":
    case "]":
      return state.depth === 1
        ? { ...next, depth: 0, open: undefined, blocks: closeOpenBlock(state) }
        : { ...next, depth: state.depth - 1 };
    case ",":
      return state.depth === 1 ? { ...next, open: undefined, blocks: closeOpenBlock(state) } : next;
    case ":": {
      const key = state.previous?.text.slice(1, -1) ?? "";
      return state.depth === 1 && state.previous?.text.startsWith('"') && isRootBlockKey(key)
        ? { ...next, open: { key, startLine: state.previous.line } }
        : next;
    }
    default:
      return next;
  }
}

export function findRootBlocks(text: string): readonly RootBlock[] {
  return closeOpenBlock(tokenize(text).reduce(step, { depth: 0, blocks: [] }));
}
