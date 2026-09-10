import { spawn } from 'node:child_process';
import { readFile, writeFile } from 'node:fs/promises';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { createServer as createViteServer } from 'vite';
import { WebSocketServer } from 'ws';

const playgroundDirectory = path.dirname(fileURLToPath(import.meta.url));
const repositoryDirectory = path.resolve(playgroundDirectory, '..');
const workspaceDirectory = path.join(playgroundDirectory, 'workspace');
const sourcePaths = ['src/main.rs', 'src/proof.rs'];
const cargoManifestPath = path.join(workspaceDirectory, 'Cargo.toml');
const port = Number(process.env.PORT ?? 5173);

const readRequestBody = request =>
  new Promise((resolve, reject) => {
    const chunks = [];
    request.on('data', chunk => chunks.push(chunk));
    request.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')));
    request.on('error', reject);
  });

const sendJson = (response, status, value) => {
  response.writeHead(status, { 'content-type': 'application/json; charset=utf-8' });
  response.end(JSON.stringify(value));
};

const runCommand = (command, args, options) =>
  new Promise(resolve => {
    const child = spawn(command, args, options);
    const stdout = [];
    const stderr = [];
    const timer = setTimeout(() => child.kill('SIGKILL'), 30_000);

    child.stdout.on('data', chunk => stdout.push(chunk));
    child.stderr.on('data', chunk => stderr.push(chunk));
    child.on('error', error => {
      clearTimeout(timer);
      resolve({ success: false, output: error.message });
    });
    child.on('close', code => {
      clearTimeout(timer);
      resolve({
        success: code === 0,
        output: Buffer.concat([...stdout, ...stderr]).toString('utf8'),
      });
    });
  });

const gitRevision = async () => {
  const result = await runCommand('git', ['rev-parse', 'HEAD'], {
    cwd: repositoryDirectory,
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  return result.success ? result.output.trim() : 'unknown';
};

const vite = await createViteServer({
  root: playgroundDirectory,
  server: { middlewareMode: true },
  appType: 'spa',
});

const server = http.createServer(async (request, response) => {
  if (request.method === 'GET' && request.url === '/api/info') {
    const [sources, revision] = await Promise.all([
      Promise.all(sourcePaths.map(async filePath => ({
        path: filePath,
        uri: pathToFileURL(path.join(workspaceDirectory, filePath)).href,
        source: await readFile(path.join(workspaceDirectory, filePath), 'utf8'),
      }))),
      gitRevision(),
    ]);
    sendJson(response, 200, {
      revision,
      files: sources,
      rootUri: pathToFileURL(workspaceDirectory).href,
    });
    return;
  }

  if (request.method === 'POST' && request.url === '/api/run') {
    try {
      const body = JSON.parse(await readRequestBody(request));
      const filesAreValid = body.files && sourcePaths.every(
        filePath => typeof body.files[filePath] === 'string',
      );
      if (!filesAreValid) {
        sendJson(response, 400, { success: false, output: 'Both Rust source files are required' });
        return;
      }
      await Promise.all(sourcePaths.map(
        filePath => writeFile(path.join(workspaceDirectory, filePath), body.files[filePath], 'utf8'),
      ));
      const result = await runCommand(
        'cargo',
        ['run', '--quiet', '--locked', '--manifest-path', cargoManifestPath],
        { cwd: workspaceDirectory, stdio: ['ignore', 'pipe', 'pipe'] },
      );
      sendJson(response, 200, result);
    } catch (error) {
      sendJson(response, 500, {
        success: false,
        output: error instanceof Error ? error.message : 'Unable to run code',
      });
    }
    return;
  }

  vite.middlewares(request, response, () => {
    response.writeHead(404);
    response.end('Not found');
  });
});

const websocketServer = new WebSocketServer({ server, path: '/lsp' });

websocketServer.on('connection', socket => {
  const analyzer = spawn('rust-analyzer', [], {
    cwd: workspaceDirectory,
    env: process.env,
    stdio: ['pipe', 'pipe', 'pipe'],
  });
  let pending = Buffer.alloc(0);

  socket.on('message', message => {
    const json = message.toString();
    const header = `Content-Length: ${Buffer.byteLength(json)}\r\n\r\n`;
    analyzer.stdin.write(header);
    analyzer.stdin.write(json);
  });

  analyzer.stdout.on('data', chunk => {
    pending = Buffer.concat([pending, chunk]);
    while (true) {
      const headerEnd = pending.indexOf('\r\n\r\n');
      if (headerEnd < 0) {
        return;
      }
      const header = pending.subarray(0, headerEnd).toString('ascii');
      const lengthMatch = /Content-Length:\s*(\d+)/i.exec(header);
      if (!lengthMatch) {
        analyzer.kill();
        socket.close(1011, 'Invalid LSP response');
        return;
      }
      const length = Number(lengthMatch[1]);
      const bodyStart = headerEnd + 4;
      const bodyEnd = bodyStart + length;
      if (pending.length < bodyEnd) {
        return;
      }
      socket.send(pending.subarray(bodyStart, bodyEnd).toString('utf8'));
      pending = pending.subarray(bodyEnd);
    }
  });

  analyzer.stderr.on('data', chunk => process.stderr.write(chunk));
  analyzer.on('close', () => socket.close());
  socket.on('close', () => analyzer.kill());
});

server.listen(port, '127.0.0.1', () => {
  process.stdout.write(`LiteLLM Rust Playground: http://localhost:${port}\n`);
});
