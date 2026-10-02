const rule = {
  meta: {
    type: "problem",
    docs: { description: "Use query readiness, not fetch activity, to render unavailable query data." },
    schema: [],
    messages: {
      readiness:
        "Query fetch flags are false while offline requests are paused. Use isQueryPending(query) for initial readiness; reserve isFetching for network activity.",
    },
  },
  create(context) {
    const services = context.sourceCode.parserServices;
    if (!services?.program || !services.esTreeNodeToTSNodeMap) {
      throw new Error("no-query-is-loading requires TypeScript parserOptions.projectService");
    }
    const checker = services.program.getTypeChecker();
    const isQuery = (node) => {
      const type = checker.getTypeAtLocation(services.esTreeNodeToTSNodeMap.get(node));
      return ["isLoading", "isPending", "isEnabled", "fetchStatus"].every((name) => type.getProperty(name));
    };
    const loadingName = /^(?:isLoading|loading|pending)$/;
    const check = (node, source) => {
      if (source && isQuery(source)) context.report({ node, messageId: "readiness" });
    };
    return {
      MemberExpression(node) {
        const name = node.computed ? node.property.value : node.property.name;
        if (name === "isLoading" || name === "isInitialLoading") check(node, node.object);
        if (name !== "isFetching") return;
        const parent = node.parent;
        if (parent.type === "VariableDeclarator" && loadingName.test(parent.id.name ?? "")) check(node, node.object);
        if (
          parent.type === "JSXExpressionContainer" &&
          parent.parent.type === "JSXAttribute" &&
          loadingName.test(parent.parent.name.name)
        )
          check(node, node.object);
      },
      Property(node) {
        if (node.parent.type !== "ObjectPattern") return;
        const name = node.key.name ?? node.key.value;
        if (name === "isLoading" || name === "isInitialLoading") check(node, node.parent);
        if (name === "isFetching" && loadingName.test(node.value.name ?? "")) check(node, node.parent);
      },
    };
  },
};

export default rule;
