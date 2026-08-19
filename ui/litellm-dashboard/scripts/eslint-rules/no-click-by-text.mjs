const TEXT_QUERIES = new Set(["getByText", "findByText", "getAllByText", "findAllByText"]);
const CLICK_METHODS = new Set(["click", "dblClick", "tripleClick", "hover"]);

const unwrapAwait = (node) => (node?.type === "AwaitExpression" ? node.argument : node);
const unwrapIndexIntoQuery = (node) => (node?.type === "MemberExpression" && node.computed ? node.object : node);

const queryNameOf = (node) => {
  const call = unwrapAwait(unwrapIndexIntoQuery(unwrapAwait(node)));
  if (call?.type !== "CallExpression") return null;
  const callee = call.callee;
  if (callee?.type === "Identifier") return TEXT_QUERIES.has(callee.name) ? callee.name : null;
  if (callee?.type === "MemberExpression" && callee.property?.type === "Identifier") {
    return TEXT_QUERIES.has(callee.property.name) ? callee.property.name : null;
  }
  return null;
};

const isClickCallee = (callee) => {
  if (callee?.type !== "MemberExpression" || callee.property?.type !== "Identifier") return false;
  if (!CLICK_METHODS.has(callee.property.name)) return false;
  const object = callee.object;
  if (object?.type === "Identifier") return /^(user|userEvent|fireEvent)$/.test(object.name);
  return object?.type === "CallExpression" || object?.type === "MemberExpression";
};

const rule = {
  meta: {
    type: "problem",
    docs: {
      description:
        "Disallow clicking an element located by its text, because text queries match hidden nodes and race a popup's open transition.",
    },
    schema: [],
    messages: {
      clickByText:
        '{{query}} matches hidden elements, so this click can land on a popup that is still closed and carries pointer-events: none, which fails intermittently. Query by role instead, e.g. getByRole("option", { name }), which waits for the element to be visible.',
    },
  },
  create(context) {
    return {
      CallExpression(node) {
        if (!isClickCallee(node.callee)) return;
        const [target] = node.arguments;
        const query = queryNameOf(target);
        if (!query) return;
        context.report({ node: target, messageId: "clickByText", data: { query } });
      },
    };
  },
};

export default rule;
