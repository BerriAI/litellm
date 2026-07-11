import noLargeInlineObjectArg from "./no-large-inline-object-arg.mjs";
import noLongConditionChain from "./no-long-condition-chain.mjs";

const plugin = {
  rules: {
    "no-large-inline-object-arg": noLargeInlineObjectArg,
    "no-long-condition-chain": noLongConditionChain,
  },
};

export default plugin;
