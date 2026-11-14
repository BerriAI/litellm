const noopHovers = (value) => {
  const tokens = value.split(/\s+/).filter(Boolean);
  const bare = new Set(tokens.filter((t) => !t.includes(":")));
  return tokens
    .filter((t) => t.startsWith("hover:"))
    .map((t) => [t, t.slice("hover:".length)])
    .filter(([, base]) => bare.has(base));
};

const rule = {
  meta: {
    type: "problem",
    docs: {
      description:
        "Disallow a hover: utility whose value is identical to the base utility in the same class string, which renders no hover feedback.",
    },
    schema: [],
    messages: {
      noop: "`{{hover}}` is identical to the base `{{base}}`, so hovering changes nothing. Give it a distinct value (e.g. `{{hover}}/80`) or drop it.",
    },
  },
  create(context) {
    const check = (node, value) => {
      if (typeof value !== "string" || !value.includes("hover:")) return;
      for (const [hover, base] of noopHovers(value)) {
        context.report({ node, messageId: "noop", data: { hover, base } });
      }
    };
    return {
      Literal(node) {
        check(node, node.value);
      },
      TemplateElement(node) {
        check(node, node.value.cooked);
      },
    };
  },
};

export default rule;
