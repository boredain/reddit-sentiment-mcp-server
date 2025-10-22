import { Match } from "./types";

export function keywordMatches(theme: string, posts: string[]): Match[] {
  const q = theme.toLowerCase();
  const matches: Match[] = [];
  posts.forEach((title, index) => {
    const text = title || "";
    const pos = text.toLowerCase().indexOf(q);
    if (pos >= 0) {
      const snippet = makeSnippet(text, pos, q.length);
      matches.push({ index, title: text, snippet });
    }
  });
  return matches;
}

function makeSnippet(text: string, start: number, len: number, radius = 120): string {
  const s = Math.max(0, start - radius);
  const e = Math.min(text.length, start + len + radius);
  const pre = s > 0 ? "…" : "";
  const post = e < text.length ? "…" : "";
  return `${pre}${text.slice(s, start)}«${text.slice(start, start + len)}»${text.slice(start + len, e)}${post}`;
}

export function toPct(count: number, total: number): number {
  return total > 0 ? count / total : 0;
}

