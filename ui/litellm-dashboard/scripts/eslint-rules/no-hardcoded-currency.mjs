const currencySymbol = /(?:^|[\s<>=(:])\$(?=\s|\d|$)/;

const rule = {
  meta: {
    type: "problem",
    docs: { description: "Use the shared money formatter instead of hard-coded dollar currency symbols" },
    schema: [],
    messages: { hardcoded: "Use the shared money formatter instead of a hard-coded dollar symbol." },
  },
  create(context) {
    const check = (node, value) => {
      if (typeof value === "string" && currencySymbol.test(value)) {
        context.report({ node, messageId: "hardcoded" });
      }
    };
    return {
      Literal(node) {
        check(node, node.value);
      },
      TemplateElement(node) {
        check(node, node.value.cooked);
      },
      JSXText(node) {
        check(node, node.value);
      },
    };
  },
};

export default rule;
