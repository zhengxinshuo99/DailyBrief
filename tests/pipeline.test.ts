import assert from "node:assert/strict";
import { test } from "node:test";

import {
  generateDailyReport,
  parseDailyReportText,
  type ArticleInput,
  type DailyReport,
} from "../lib/ai/pipeline";
import type { LlmRunner } from "../lib/ai/llm";

const validReport: DailyReport = {
  hero_headline: "今日重要资讯汇总",
  daily_overview: "技术、财经与国际新闻均有值得关注的新进展。",
  tech_briefs: [
    {
      title: "技术新闻",
      url: "https://example.com/tech",
      source: "Tech Source",
      summary: "技术新闻摘要。",
      importance: 8,
    },
  ],
  finance_briefs: [
    {
      title: "财经新闻",
      url: "https://example.com/finance",
      source: "Finance Source",
      summary: "财经新闻摘要。",
      importance: 7,
    },
  ],
  politics_briefs: [
    {
      title: "国际新闻",
      url: "https://example.com/politics",
      source: "World Source",
      summary: "国际新闻摘要。",
      importance: 6,
    },
  ],
  editor_note: "本期内容来自多源公开信息。",
  keywords: ["技术", "财经", "国际"],
};

const articles: ArticleInput[] = [
  {
    sourceId: "tech-source",
    source: "Tech Source",
    title: "技术新闻",
    url: "https://example.com/tech",
    excerpt:
      "技术新闻的原始摘要，包含足够多的上下文用于验证重试时会逐步缩短输入。".repeat(
        12,
      ),
    summary: "已经生成的技术摘要。",
    category: "tech",
  },
  {
    sourceId: "finance-source",
    source: "Finance Source",
    title: "财经新闻",
    url: "https://example.com/finance",
    excerpt:
      "财经新闻的原始摘要，包含足够多的上下文用于验证重试时会逐步缩短输入。".repeat(
        12,
      ),
    category: "finance",
  },
  {
    sourceId: "world-source",
    source: "World Source",
    title: "国际新闻",
    url: "https://example.com/politics",
    excerpt:
      "国际新闻的原始摘要，包含足够多的上下文用于验证重试时会逐步缩短输入。".repeat(
        12,
      ),
    category: "politics",
  },
];

test("parseDailyReportText accepts a complete report", () => {
  assert.deepEqual(parseDailyReportText(JSON.stringify(validReport)), validReport);
});

test("parseDailyReportText keeps jsonrepair recovery for complete reports", () => {
  const repairable = JSON.stringify(validReport).replace(/}$/, ",}");
  assert.deepEqual(parseDailyReportText(repairable), validReport);
});

test("parseDailyReportText rejects a repaired but incomplete report", () => {
  assert.throws(
    () =>
      parseDailyReportText(
        '{"hero_headline":"partial","daily_overview":"cut off"',
      ),
    /incomplete|invalid|missing/i,
  );
});

test("parseDailyReportText rejects an empty brief list when candidates exist", () => {
  assert.throws(
    () =>
      parseDailyReportText(
        JSON.stringify({ ...validReport, tech_briefs: [] }),
        new Set(["tech"]),
      ),
    /tech_briefs.*empty/i,
  );
});

test("generateDailyReport retries an invalid response and accepts the next valid one", async () => {
  let calls = 0;
  const runner: LlmRunner = async () => ({
    text: ++calls === 1 ? "" : JSON.stringify(validReport),
    durationMs: 1,
  });

  const result = await generateDailyReport(articles, runner);
  assert.equal(calls, 2);
  assert.deepEqual(result.report, validReport);
});

test("generateDailyReport shrinks retries and returns a deterministic fallback", async () => {
  const promptLengths: number[] = [];
  const runner: LlmRunner = async (options) => {
    promptLengths.push(options.userPrompt.length);
    return { text: "", durationMs: 1 };
  };

  const { report } = await generateDailyReport(articles, runner);
  assert.equal(promptLengths.length, 3);
  assert(promptLengths[0] > promptLengths[1]);
  assert(promptLengths[1] > promptLengths[2]);
  assert.equal(report.tech_briefs[0].url, articles[0].url);
  assert.equal(report.tech_briefs[0].summary, articles[0].summary);
  assert.equal(report.finance_briefs[0].url, articles[1].url);
  assert.equal(report.politics_briefs[0].url, articles[2].url);
  assert(report.hero_headline.length > 0);
  assert(report.daily_overview.length > 0);
  assert(report.editor_note.length > 0);
});

test("generateDailyReport does not hide authentication failures", async () => {
  let calls = 0;
  const runner: LlmRunner = async () => {
    calls += 1;
    throw new Error("401 Unauthorized");
  };

  await assert.rejects(generateDailyReport(articles, runner), /401 Unauthorized/);
  assert.equal(calls, 1);
});
