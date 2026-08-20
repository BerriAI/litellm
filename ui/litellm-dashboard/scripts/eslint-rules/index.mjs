import noLargeInlineObjectArg from "./no-large-inline-object-arg.mjs";
import noLongConditionChain from "./no-long-condition-chain.mjs";
import noComplexJsxArrow from "./no-complex-jsx-arrow.mjs";
import filenamePascalCase from "./filename-pascal-case.mjs";
import noNoopHoverVariant from "./no-noop-hover-variant.mjs";

const plugin = {
  rules: {
    "no-large-inline-object-arg": noLargeInlineObjectArg,
    "no-long-condition-chain": noLongConditionChain,
    "no-complex-jsx-arrow": noComplexJsxArrow,
    "filename-pascal-case": filenamePascalCase,
    "no-noop-hover-variant": noNoopHoverVariant,
  },
};

export default plugin;
