const message = "Use React Query (@tanstack/react-query) for data fetching instead of a raw fetch().";

const isInsideReactQueryFn = (node) => {
  for (let current = node.parent; current; current = current.parent) {
    if (current.type === "Property" && !current.computed) {
      const keyName = current.key.name ?? current.key.value;
      if (keyName === "queryFn" || keyName === "mutationFn") {
        return true;
      }
    }
  }
  return false;
};

export default {
  meta: {
    type: "problem",
    docs: {
      description: "Disallow raw fetch() unless it is lexically inside a React Query queryFn/mutationFn",
    },
    messages: { bareFetch: message },
    schema: [],
  },
  create(context) {
    return {
      "CallExpression[callee.name='fetch']"(node) {
        if (!isInsideReactQueryFn(node)) {
          context.report({ node, messageId: "bareFetch" });
        }
      },
    };
  },
};
