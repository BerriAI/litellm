const SELECTOR_REFERENCE = /\.(?:ant|anticon)-[a-z0-9-]+/;
const BARE_CLASS_REFERENCE = /^(?:ant|anticon)-[a-z0-9-]+$/;
const CLASS_ASSERTION_CALLEES = new Set(["toHaveClass", "contains", "toContain"]);

const isClassAssertionArgument = (node) => {
  const call = node.parent;
  if (call?.type !== "CallExpression" || !call.arguments.includes(node)) return false;
  const callee = call.callee;
  return callee?.type === "MemberExpression" && CLASS_ASSERTION_CALLEES.has(callee.property?.name);
};

const rule = {
  meta: {
    type: "problem",
    docs: {
      description:
        "Disallow locating or asserting on antd's internal CSS classes in tests; query by role, label or text instead.",
    },
    schema: [],
    messages: {
      antdClass:
        'Test depends on antd internal class "{{value}}". Query by role, label or text (getByLabelText, getByRole("combobox"), getByTitle) so the test survives the shadcn migration.',
    },
  },
  create(context) {
    const report = (node, value) => {
      if (typeof value !== "string") return;
      const matches =
        SELECTOR_REFERENCE.test(value) || (BARE_CLASS_REFERENCE.test(value) && isClassAssertionArgument(node));
      if (!matches) return;
      context.report({ node, messageId: "antdClass", data: { value } });
    };

    return {
      Literal(node) {
        report(node, node.value);
      },
      TemplateElement(node) {
        report(node, node.value.cooked);
      },
    };
  },
};

export default rule;
