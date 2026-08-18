import assert from "node:assert/strict";
import { test } from "node:test";

import { fetchLinuxDo } from "../lib/sources/linuxdo";

const VALID_RSS = `<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Linux.do</title>
    <link>https://linux.do/</link>
    <description>Latest topics</description>
    <item>
      <title>Useful topic</title>
      <link>https://linux.do/t/topic/1</link>
      <description><![CDATA[<p>Useful description</p>]]></description>
      <pubDate>Mon, 18 Aug 2026 00:00:00 GMT</pubDate>
    </item>
  </channel>
</rss>`;

test("Linux.do uses the configured primary URL", async () => {
  const urls: string[] = [];
  const items = await fetchLinuxDo("linuxdo", {
    primaryUrl: "https://mirror.example/top.rss",
    fallbackUrl: "https://mirror.example/latest.rss",
    fetchText: async (url) => {
      urls.push(url);
      return VALID_RSS;
    },
  });

  assert.deepEqual(urls, ["https://mirror.example/top.rss"]);
  assert.equal(items.length, 1);
  assert.equal(items[0].title, "Useful topic");
});

test("Linux.do falls back after a malformed primary response", async () => {
  const urls: string[] = [];
  const items = await fetchLinuxDo("linuxdo", {
    primaryUrl: "https://linux.do/top.rss?period=daily",
    fallbackUrl: "https://linux.do/latest.rss",
    fetchText: async (url) => {
      urls.push(url);
      return urls.length === 1 ? "<rss><broken>" : VALID_RSS;
    },
  });

  assert.equal(items.length, 1);
  assert.deepEqual(urls, [
    "https://linux.do/top.rss?period=daily",
    "https://linux.do/latest.rss",
  ]);
});

test("Linux.do combines endpoint diagnostics without leaking response bodies", async () => {
  const privateBody = "<html>PRIVATE-UPSTREAM-BODY</html>";
  await assert.rejects(
    fetchLinuxDo("linuxdo", {
      primaryUrl: "https://linux.do/top.rss?period=daily",
      fallbackUrl: "https://linux.do/latest.rss",
      fetchText: async () => privateBody,
    }),
    (error: unknown) => {
      assert(error instanceof Error);
      assert.match(error.message, /primary/i);
      assert.match(error.message, /fallback/i);
      assert.match(error.message, /html|non-rss/i);
      assert.doesNotMatch(error.message, /PRIVATE-UPSTREAM-BODY/);
      return true;
    },
  );
});
