const isAntd = (source) => typeof source === "string" && /^antd(\/|$)/.test(source);

const rule = {
  meta: {
    type: "suggestion",
    docs: {
      description: "Discourage antd imports; antd is being phased out in favor of shadcn/ui primitives.",
    },
    schema: [],
    messages: {
      antdImport: "antd is being phased out; build new UI with shadcn/ui primitives instead of adding antd imports.",
    },
  },
  create(context) {
    const report = (node) => context.report({ node, messageId: "antdImport" });
    const checkSource = (node) => {
      if (node?.source && isAntd(node.source.value)) report(node.source);
    };
    return {
      ImportDeclaration: checkSource,
      ExportNamedDeclaration: checkSource,
      ExportAllDeclaration: checkSource,
      ImportExpression(node) {
        if (node.source?.type === "Literal" && isAntd(node.source.value)) report(node.source);
      },
      "CallExpression[callee.name='require']"(node) {
        const arg = node.arguments[0];
        if (arg?.type === "Literal" && isAntd(arg.value)) report(arg);
      },
    };
  },
};

export default rule;
