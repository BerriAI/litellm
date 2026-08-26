const AD_HOC_Z = /^(?:[\w-]+:)*-?z-(?:\d+|\[[^\]]*\]|\([^)]*\))$/;
const POPUP_Z = /^(?:[\w-]+:)*z-popup$/;

const classify = (token, allowPopupLayer) => {
  if (AD_HOC_Z.test(token)) return "adHoc";
  if (!allowPopupLayer && POPUP_Z.test(token)) return "popupReserved";
  return null;
};

const offendingTokens = (value, allowPopupLayer) =>
  value
    .split(/\s+/)
    .filter(Boolean)
    .map((token) => ({ token, messageId: classify(token, allowPopupLayer) }))
    .filter(({ messageId }) => messageId !== null);

const propertyName = (key) => {
  if (key.type === "Identifier") return key.name;
  if (key.type === "Literal" && typeof key.value === "string") return key.value;
  return null;
};

const rule = {
  meta: {
    type: "problem",
    docs: {
      description:
        "Disallow hand-picked z-index values (numeric or arbitrary z-* classes, inline zIndex styles). Use the named scale defined in src/app/globals.css so nothing can stack above the portalled popup layer.",
    },
    schema: [
      {
        type: "object",
        properties: { allowPopupLayer: { type: "boolean" } },
        additionalProperties: false,
      },
    ],
    messages: {
      adHoc:
        "`{{token}}` is a hand-picked z-index. Use the scale from globals.css: z-raised, z-chrome, z-sticky, z-sticky-pinned, z-floating, z-overlay (z-popup is reserved for portalled primitives).",
      popupReserved:
        "`{{token}}` is reserved for the portalled primitives in src/components/ui. Page content must stay below the popup layer; use z-overlay or lower.",
      inlineZIndex:
        "Inline `zIndex` styles bypass the z-index scale. Use a class from globals.css (z-raised, z-chrome, z-sticky, z-sticky-pinned, z-floating, z-overlay) instead.",
    },
  },
  create(context) {
    const allowPopupLayer = context.options[0]?.allowPopupLayer ?? false;
    const check = (node, value) => {
      if (typeof value !== "string" || !value.includes("z-")) return;
      for (const { token, messageId } of offendingTokens(value, allowPopupLayer)) {
        context.report({ node, messageId, data: { token } });
      }
    };
    return {
      Literal(node) {
        check(node, node.value);
      },
      TemplateElement(node) {
        check(node, node.value.cooked);
      },
      Property(node) {
        const name = propertyName(node.key);
        if (name === "zIndex" || name === "z-index") {
          context.report({ node, messageId: "inlineZIndex" });
        }
      },
    };
  },
};

export default rule;
