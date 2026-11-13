const assert = require("node:assert/strict");
const { test } = require("node:test");
const { restoreHandwrittenDescription, keepHandwrittenDescription } = require("./keep_handwritten_description.cjs");

const heading = "## Handwritten 1-3 sentence description";
const placeholder = "[insert handwritten description here]";
const details = "<details>\n<summary>Details</summary>\n\n## TLDR\nNew details\n</details>";
const body = (text) => `${heading}\n\n${text}\n\n${details}`;

test("wipe to placeholder restores text and preserves new details", () => {
  assert.equal(
    restoreHandwrittenDescription(
      body("Keep the author's words."),
      body(`<!-- template instructions -->\n${placeholder}`),
    ),
    body("Keep the author's words."),
  );
});

test("removed heading is restored at the top without changing the new body", () => {
  assert.equal(restoreHandwrittenDescription(body("Original words"), details), body("Original words"));
});

test("real text replaced by different real text is left alone", () => {
  assert.equal(restoreHandwrittenDescription(body("Original"), body("Human revision")), null);
});

test("placeholder to placeholder is a no-op", () => {
  assert.equal(restoreHandwrittenDescription(body(placeholder), body(placeholder)), null);
});

test("null old body is a no-op", () => {
  assert.equal(restoreHandwrittenDescription(null, body(placeholder)), null);
});

test("empty and comment-only sections are restored", () => {
  for (const text of ["", "<!-- instructions\n## hidden heading\n-->"]) {
    assert.equal(restoreHandwrittenDescription(body("Original"), body(text)), body("Original"));
  }
});

test("old comments and details are not mistaken for handwritten text", () => {
  for (const text of ["", "<!-- real-looking words -->", placeholder]) {
    assert.equal(restoreHandwrittenDescription(body(text), body("")), null);
  }
});

test("restoration excludes old comments and stops at the next section", () => {
  assert.equal(
    restoreHandwrittenDescription(
      `${heading}\n<!-- instructions -->\nOriginal\n\n## Other\nOld details`,
      `${heading}\n\n${placeholder}\n\n## Other\nNew details`,
    ),
    `${heading}\n\nOriginal\n\n## Other\nNew details`,
  );
});

test("null new body restores the heading and text", () => {
  assert.equal(restoreHandwrittenDescription(body("Original"), null), `${heading}\n\nOriginal\n\n`);
});

function harness(currentBody, update) {
  return {
    context: {
      repo: { owner: "owner", repo: "repo" },
      payload: {
        changes: { body: { from: body("Original") } },
        pull_request: { number: 123, body: body(placeholder) },
      },
    },
    github: {
      rest: {
        pulls: {
          get: async () => ({ data: { body: currentBody } }),
          update,
        },
      },
    },
  };
}

test("API restores the event body once and restoration cannot loop", async () => {
  const updates = [];
  await keepHandwrittenDescription(harness(body(placeholder), async (params) => updates.push(params)));
  assert.deepEqual(updates, [{ owner: "owner", repo: "repo", pull_number: 123, body: body("Original") }]);
  assert.equal(restoreHandwrittenDescription(body(placeholder), updates[0].body), null);
});

test("a newer edit is not overwritten", async () => {
  await keepHandwrittenDescription(harness(body("Newer human edit"), async () => assert.fail("unexpected update")));
});

test("API failures are surfaced", async () => {
  await assert.rejects(
    keepHandwrittenDescription(
      harness(body(placeholder), async () => {
        throw new Error("API rejected update");
      }),
    ),
    /API rejected update/,
  );
});
