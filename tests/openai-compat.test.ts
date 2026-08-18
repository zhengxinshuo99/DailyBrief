import assert from "node:assert/strict";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { test } from "node:test";

import {
  PRESETS,
  runOpenAICompat,
} from "../lib/ai/backends/openai-compat";

interface MockCall {
  body: Record<string, unknown>;
}

async function readJson(req: IncomingMessage): Promise<Record<string, unknown>> {
  const chunks: Buffer[] = [];
  for await (const chunk of req) chunks.push(Buffer.from(chunk));
  return JSON.parse(Buffer.concat(chunks).toString("utf8")) as Record<string, unknown>;
}

async function withMockDeepSeek<T>(
  responseBody: Record<string, unknown>,
  run: (calls: MockCall[]) => Promise<T>,
): Promise<T> {
  const calls: MockCall[] = [];
  const server = createServer(async (req: IncomingMessage, res: ServerResponse) => {
    calls.push({ body: await readJson(req) });
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify(responseBody));
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));

  const address = server.address();
  assert(address && typeof address === "object");
  const oldKey = process.env.DEEPSEEK_API_KEY;
  const oldBaseUrl = process.env.DEEPSEEK_BASE_URL;
  const oldModel = process.env.LLM_MODEL;
  process.env.DEEPSEEK_API_KEY = "test-key";
  process.env.DEEPSEEK_BASE_URL = `http://127.0.0.1:${address.port}/v1`;
  process.env.LLM_MODEL = "test-model";

  try {
    return await run(calls);
  } finally {
    server.closeAllConnections();
    await new Promise<void>((resolve, reject) =>
      server.close((error) => (error ? reject(error) : resolve())),
    );
    if (oldKey === undefined) delete process.env.DEEPSEEK_API_KEY;
    else process.env.DEEPSEEK_API_KEY = oldKey;
    if (oldBaseUrl === undefined) delete process.env.DEEPSEEK_BASE_URL;
    else process.env.DEEPSEEK_BASE_URL = oldBaseUrl;
    if (oldModel === undefined) delete process.env.LLM_MODEL;
    else process.env.LLM_MODEL = oldModel;
  }
}

function response(content: string | null, finishReason: string) {
  return {
    id: "mock-response",
    object: "chat.completion",
    created: 0,
    model: "test-model",
    choices: [
      {
        index: 0,
        message: { role: "assistant", content },
        finish_reason: finishReason,
      },
    ],
    usage: { prompt_tokens: 10, completion_tokens: 20, total_tokens: 30 },
  };
}

const options = { systemPrompt: "Return JSON", userPrompt: "Input" };

test("DeepSeek requests retain the 8192-token output budget", async () => {
  await withMockDeepSeek(response('{"ok":true}', "stop"), async (calls) => {
    const result = await runOpenAICompat(options, PRESETS.deepseek);
    assert.equal(result.text, '{"ok":true}');
    assert.equal(calls.length, 1);
    assert.equal(calls[0].body.max_tokens, 8192);
    assert.deepEqual(calls[0].body.response_format, { type: "json_object" });
    assert.deepEqual(calls[0].body.thinking, { type: "disabled" });
  });
});

test("DeepSeek rejects a response truncated by the token limit", async () => {
  await withMockDeepSeek(response('{"partial":', "length"), async () => {
    await assert.rejects(
      runOpenAICompat(options, PRESETS.deepseek),
      /truncated|finish_reason=length/i,
    );
  });
});

test("DeepSeek rejects a blank response body", async () => {
  await withMockDeepSeek(response(null, "stop"), async () => {
    await assert.rejects(
      runOpenAICompat(options, PRESETS.deepseek),
      /empty|blank|content/i,
    );
  });
});

test("DeepSeek rejects an interrupted inference response", async () => {
  await withMockDeepSeek(
    response('{"partial":true}', "insufficient_system_resource"),
    async () => {
      await assert.rejects(
        runOpenAICompat(options, PRESETS.deepseek),
        /incomplete|insufficient_system_resource|interrupted/i,
      );
    },
  );
});
