const OPEN_TAG = "<think>";
const CLOSE_TAG = "</think>";

export interface ThinkTagState {
  readonly insideThink: boolean;
  readonly pending: string;
}

export interface ThinkTagSplit {
  readonly state: ThinkTagState;
  readonly text: string;
  readonly reasoning: string;
}

export const initialThinkTagState: ThinkTagState = { insideThink: false, pending: "" };

function heldBackLength(buffer: string, tag: string): number {
  const max = Math.min(buffer.length, tag.length - 1);
  for (let len = max; len > 0; len--) {
    if (tag.startsWith(buffer.slice(buffer.length - len))) return len;
  }
  return 0;
}

export function splitThinkTags(state: ThinkTagState, delta: string): ThinkTagSplit {
  const consume = (
    buffer: string,
    insideThink: boolean,
    text: string,
    reasoning: string,
  ): { pending: string; insideThink: boolean; text: string; reasoning: string } => {
    const tag = insideThink ? CLOSE_TAG : OPEN_TAG;
    const index = buffer.indexOf(tag);
    if (index !== -1) {
      const before = buffer.slice(0, index);
      return consume(
        buffer.slice(index + tag.length),
        !insideThink,
        insideThink ? text : text + before,
        insideThink ? reasoning + before : reasoning,
      );
    }
    const held = heldBackLength(buffer, tag);
    const emitted = buffer.slice(0, buffer.length - held);
    return {
      pending: buffer.slice(buffer.length - held),
      insideThink,
      text: insideThink ? text : text + emitted,
      reasoning: insideThink ? reasoning + emitted : reasoning,
    };
  };

  const result = consume(state.pending + delta, state.insideThink, "", "");
  return {
    state: { insideThink: result.insideThink, pending: result.pending },
    text: result.text,
    reasoning: result.reasoning,
  };
}

export function flushThinkTags(state: ThinkTagState): ThinkTagSplit {
  return {
    state: initialThinkTagState,
    text: state.insideThink ? "" : state.pending,
    reasoning: state.insideThink ? state.pending : "",
  };
}
