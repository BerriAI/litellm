import { utilityOf } from "./tailwind-utility.mjs";

const ARBITRARY_SCALE = /^(?:text|tracking|leading|rounded(?:-[a-z]+)?|border(?:-[a-z]+)?)-\[/;
const ARBITRARY_PROPERTY = /^\[[a-z-]+:/;

const isOffending = (token) => {
  const utility = utilityOf(token);
  return ARBITRARY_SCALE.test(utility) || ARBITRARY_PROPERTY.test(utility);
};

const rule = {
  meta: {
    type: "problem",
    docs: {
      description:
        "Disallow arbitrary font size, tracking, leading, radius and border values, and arbitrary CSS properties. Use the theme scale so surfaces share one type and shape system.",
    },
    schema: [],
    messages: {
      arbitrary:
        "`{{token}}` bypasses the theme scale. Use a scale utility (text-xs/sm, leading-*, tracking-*, rounded-sm/md/lg, border/border-2) instead.",
    },
  },
  create(context) {
    const check = (node, value) => {
      if (typeof value !== "string" || !value.includes("[")) return;
      for (const token of value.split(/\s+/).filter(isOffending)) {
        context.report({ node, messageId: "arbitrary", data: { token } });
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
