const DEFAULT_MAX_STATEMENTS = 2;

const isJsxAttributeValue = (node) => {
  const parent = node.parent;
  if (parent == null) return false;
  return parent.type === "JSXExpressionContainer" && parent.parent?.type === "JSXAttribute";
};

const rule = {
  meta: {
    type: "suggestion",
    docs: {
      description:
        "Disallow arrow functions with block bodies over a few statements passed inline as JSX attributes; extract them into a named handler.",
    },
    schema: [
      {
        type: "object",
        properties: { maxStatements: { type: "integer", minimum: 1 } },
        additionalProperties: false,
      },
    ],
    messages: {
      tooComplex: "Inline JSX arrow handler has {{count}} statements; extract it into a named function (max {{max}}).",
    },
  },
  create(context) {
    const maxStatements = context.options[0]?.maxStatements ?? DEFAULT_MAX_STATEMENTS;
    return {
      ArrowFunctionExpression(node) {
        if (node.body.type !== "BlockStatement") return;
        if (!isJsxAttributeValue(node)) return;
        const count = node.body.body.length;
        if (count <= maxStatements) return;
        context.report({ node, messageId: "tooComplex", data: { count, max: maxStatements } });
      },
    };
  },
};

export default rule;
