const vscode = require('vscode');
function activate(context) {
  context.subscriptions.push(vscode.commands.registerCommand('lit7449.start', () => vscode.commands.executeCommand('workbench.mcp.startServer', '*')));
  const output = vscode.window.createOutputChannel('LIT-7449 Verification');
  context.subscriptions.push(output, vscode.commands.registerCommand('lit7449.verifyTools', async () => {
    output.clear();
    output.show();
    const tools = vscode.lm.tools.filter(t => t.name.includes('microsoft_docs_search'));
    output.appendLine(`VS Code discovered ${tools.length} matching documentation search tool(s)`);
    if (tools.length !== 1) {
      output.appendLine(JSON.stringify(vscode.lm.tools.map(t => t.name)));
      throw new Error('Expected one Microsoft Learn search tool');
    }
    output.appendLine(`Invoking ${tools[0].name} through vscode.lm.invokeTool`);
    try {
      const result = await vscode.lm.invokeTool(tools[0].name, {input: {query: 'Azure resource groups'}, toolInvocationToken: undefined});
      const text = result.content.filter(c => c instanceof vscode.LanguageModelTextPart).map(c => c.value).join('\n');
      if (!text) throw new Error('No text returned');
      output.appendLine('PASS: actual VS Code MCP tool call returned documentation');
      output.appendLine(text);
    } catch (error) {
      output.appendLine(`FAIL: ${error.message}`);
      throw error;
    }
  }));
}
exports.activate = activate;
