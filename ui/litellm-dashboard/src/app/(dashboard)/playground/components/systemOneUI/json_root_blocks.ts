export type RootBlockKey = "model" | "state" | "questions";

export interface RootBlock {
  key: RootBlockKey;
  startLine: number;
  endLine: number;
}

export const ROOT_BLOCK_STYLES: Record<RootBlockKey, { band: string; accent: string }> = {
  model: { band: "bg-blue-500/10 border-blue-500/60", accent: "border-l-blue-500/60" },
  state: { band: "bg-green-500/10 border-green-500/60", accent: "border-l-green-500/60" },
  questions: { band: "bg-orange-500/10 border-orange-500/60", accent: "border-l-orange-500/60" },
};

const isRootBlockKey = (key: string): key is RootBlockKey => Object.hasOwn(ROOT_BLOCK_STYLES, key);

export function findRootBlocks(text: string): RootBlock[] {
  const blocks: RootBlock[] = [];
  let depth = 0;
  let line = 0;
  let lastContentLine = 0;
  let inString = false;
  let escaped = false;
  let stringStart = 0;
  let stringStartLine = 0;
  let lastString = { value: "", line: 0 };
  let open: { key: RootBlockKey; startLine: number } | undefined;

  const close = (endLine: number) => {
    if (open) {
      blocks.push({ ...open, endLine });
      open = undefined;
    }
  };

  for (let index = 0; index < text.length; index += 1) {
    const character = text[index];
    if (character === "\n") {
      line += 1;
      continue;
    }
    if (inString) {
      lastContentLine = line;
      if (escaped) {
        escaped = false;
      } else if (character === "\\") {
        escaped = true;
      } else if (character === '"') {
        inString = false;
        lastString = { value: text.slice(stringStart + 1, index), line: stringStartLine };
      }
      continue;
    }
    if (/\s/.test(character)) {
      continue;
    }

    const previousContentLine = lastContentLine;
    lastContentLine = line;
    if (character === '"') {
      inString = true;
      stringStart = index;
      stringStartLine = line;
    } else if (character === ":" && depth === 1 && isRootBlockKey(lastString.value)) {
      open = { key: lastString.value, startLine: lastString.line };
    } else if (character === "," && depth === 1) {
      close(previousContentLine);
    } else if (character === "{" || character === "[") {
      depth += 1;
    } else if (character === "}" || character === "]") {
      depth -= 1;
      if (depth === 0) {
        close(previousContentLine);
      }
    }
  }
  close(lastContentLine);
  return blocks;
}
