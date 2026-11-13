const HEADING = "## Handwritten 1-3 sentence description";
const PLACEHOLDER = "[insert handwritten description here]";

function withoutComments(body) {
  return body.replace(/<!--[\s\S]*?(?:-->|$)/g, (comment) => comment.replace(/[^\r\n]/g, " "));
}

function section(body) {
  const visible = withoutComments(body);
  const heading = /^## Handwritten 1-3 sentence description[ \t]*\r?$/m.exec(visible);
  if (!heading) return null;
  const start = heading.index + heading[0].length;
  const boundary = /^(?:#{1,2}[ \t]+|[ \t]*<\/?details\b)/im.exec(visible.slice(start));
  const end = boundary ? start + boundary.index : body.length;
  return { start, end, text: visible.slice(start, end).trim() };
}

function hasRealText(text) {
  return text.replaceAll(PLACEHOLDER, "").trim().length > 0;
}

function restoreHandwrittenDescription(oldBody, newBody) {
  const oldSection = section(oldBody ?? "");
  if (!oldSection || !hasRealText(oldSection.text)) return null;
  const body = newBody ?? "";
  const newSection = section(body);
  if (newSection && hasRealText(newSection.text)) return null;
  if (!newSection) return `${HEADING}\n\n${oldSection.text}\n\n${body}`;
  return `${body.slice(0, newSection.start)}\n\n${oldSection.text}\n\n${body.slice(newSection.end)}`;
}

async function keepHandwrittenDescription({ github, context }) {
  const { pull_request: pullRequest, changes } = context.payload;
  const restored = restoreHandwrittenDescription(changes?.body?.from, pullRequest.body);
  if (restored === null || restored === pullRequest.body) return;
  const params = { ...context.repo, pull_number: pullRequest.number };
  const { data: current } = await github.rest.pulls.get(params);
  if (current.body !== pullRequest.body) return;
  await github.rest.pulls.update({ ...params, body: restored });
}

module.exports = { restoreHandwrittenDescription, keepHandwrittenDescription };
