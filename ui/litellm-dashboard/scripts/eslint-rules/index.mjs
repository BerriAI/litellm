import noLargeInlineObjectArg from "./no-large-inline-object-arg.mjs";
import noLongConditionChain from "./no-long-condition-chain.mjs";
import noComplexJsxArrow from "./no-complex-jsx-arrow.mjs";
import filenamePascalCase from "./filename-pascal-case.mjs";
import noNoopHoverVariant from "./no-noop-hover-variant.mjs";
import noAdHocZIndex from "./no-ad-hoc-z-index.mjs";
import noArbitraryDesignValue from "./no-arbitrary-design-value.mjs";

const plugin = {
  rules: {
    "no-large-inline-object-arg": noLargeInlineObjectArg,
    "no-long-condition-chain": noLongConditionChain,
    "no-complex-jsx-arrow": noComplexJsxArrow,
    "filename-pascal-case": filenamePascalCase,
    "no-noop-hover-variant": noNoopHoverVariant,
    "no-ad-hoc-z-index": noAdHocZIndex,
    "no-arbitrary-design-value": noArbitraryDesignValue,
  },
};

export default plugin;
