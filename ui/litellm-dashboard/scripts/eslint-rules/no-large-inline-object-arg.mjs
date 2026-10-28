const DEFAULT_MIN_PROPERTIES = 4;

const isArgumentOf = (node) => {
  const parent = node.parent;
  if (parent == null) return false;
  if (parent.type !== "CallExpression" && parent.type !== "NewExpression") return false;
  return parent.arguments.includes(node);
};

const rule = {
  meta: {
    type: "suggestion",
    docs: {
      description:
        "Disallow passing a large object literal inline as a call argument; assign it to a named variable first.",
    },
    schema: [
      {
        type: "object",
        properties: { minProperties: { type: "integer", minimum: 1 } },
        additionalProperties: false,
      },
    ],
    messages: {
      tooLarge:
        "Object literal with {{count}} properties passed inline as an argument; assign it to a named variable first.",
    },
  },
  create(context) {
    const minProperties = context.options[0]?.minProperties ?? DEFAULT_MIN_PROPERTIES;
    return {
      ObjectExpression(node) {
        if (!isArgumentOf(node)) return;
        if (node.properties.length < minProperties) return;
        context.report({ node, messageId: "tooLarge", data: { count: node.properties.length } });
      },
    };
  },
};

export default rule;
