const DEFAULT_MIN_CONDITIONS = 4;

const countConditions = (node) =>
  node.type === "LogicalExpression" ? countConditions(node.left) + countConditions(node.right) : 1;

const rule = {
  meta: {
    type: "suggestion",
    docs: {
      description:
        "Disallow logical expressions that combine many conditions; extract the condition into a named boolean.",
    },
    schema: [
      {
        type: "object",
        properties: { minConditions: { type: "integer", minimum: 2 } },
        additionalProperties: false,
      },
    ],
    messages: {
      tooMany: "Boolean expression combines {{count}} conditions; extract it into a named variable.",
    },
  },
  create(context) {
    const minConditions = context.options[0]?.minConditions ?? DEFAULT_MIN_CONDITIONS;
    return {
      LogicalExpression(node) {
        if (node.parent?.type === "LogicalExpression") return;
        const count = countConditions(node);
        if (count < minConditions) return;
        context.report({ node, messageId: "tooMany", data: { count } });
      },
    };
  },
};

export default rule;
