import React, { useMemo } from "react";
import { createRoot } from "react-dom/client";
import { useToolOutputs, useWidgetState } from "./sdk-hooks";
import { Payload, Category, ThemeItem, InsightItem, SummaryBullet } from "./types";

type ToolEnvelope = { text?: string; structuredContent?: any; content?: any } | Payload;

function extractStructured(env: any): any | undefined {
  console.log('[extractStructured] input:', env);

  if (!env) return undefined;

  // Direct structuredContent
  if (env.structuredContent) {
    console.log('[extractStructured] found via structuredContent');
    return env.structuredContent;
  }

  // Wrapped under result
  if (env.result?.structuredContent) {
    console.log('[extractStructured] found via result.structuredContent');
    return env.result.structuredContent;
  }

  // Array of outputs
  if (env.outputs?.[0]?.structuredContent) {
    console.log('[extractStructured] found via outputs[0].structuredContent');
    return env.outputs[0].structuredContent;
  }

  // ChatGPT may pass content array directly
  if (Array.isArray(env?.content) && env.content.length > 0) {
    try {
      const first = env.content[0];
      if (first?.text) {
        const parsed = JSON.parse(first.text);
        console.log('[extractStructured] found via content[0].text (parsed)');
        return parsed;
      }
    } catch (e) {
      console.log('[extractStructured] failed to parse content[0].text:', e);
    }
  }

  // ChatGPT Apps SDK may pass the data directly as an array
  if (Array.isArray(env) && env.length > 0) {
    const first = env[0];
    if (first?.structuredContent) {
      console.log('[extractStructured] found via array[0].structuredContent');
      return first.structuredContent;
    }
    if (first?.content?.[0]?.text) {
      try {
        const parsed = JSON.parse(first.content[0].text);
        console.log('[extractStructured] found via array[0].content[0].text (parsed)');
        return parsed;
      } catch (e) {
        console.log('[extractStructured] failed to parse array[0].content[0].text:', e);
      }
    }
  }

  // Deep search for a payload-like object
  try {
    const seen = new Set<any>();
    const stack = [env];
    while (stack.length) {
      const cur = stack.pop();
      if (!cur || typeof cur !== "object" || seen.has(cur)) continue;
      seen.add(cur);
      if (cur.overview && cur.business_insights && cur.posts) {
        console.log('[extractStructured] found via deep search');
        return cur;
      }
      // Crawl arrays and objects
      if (Array.isArray(cur)) {
        for (const it of cur) stack.push(it);
      } else {
        for (const k of Object.keys(cur)) stack.push((cur as any)[k]);
      }
    }
  } catch (e) {
    console.log('[extractStructured] deep search failed:', e);
  }

  console.log('[extractStructured] no structured data found');
  return undefined;
}

function parsePayload(env: ToolEnvelope | undefined): Payload | null {
  if (!env) return null;
  // Prefer structuredContent if present
  const structured = extractStructured(env);
  if (structured) return structured as Payload;
  // Fallback: some servers wrap JSON as a string in { text }
  const candidate = (env as any).text ? (env as any).text : env;
  try {
    return typeof candidate === "string" ? (JSON.parse(candidate) as Payload) : (candidate as Payload);
  } catch {
    return null;
  }
}

export default function App() {
  const toolOutputs = useToolOutputs<ToolEnvelope>();
  const [drawer, setDrawer] = useWidgetState<{ theme?: string; category?: Category; open?: boolean; source_indices?: number[] }>({});

  const payload = useMemo(() => parsePayload(toolOutputs), [toolOutputs]);

  console.log('[App] toolOutputs:', toolOutputs);
  console.log('[App] window.openai:', (window as any).openai);
  console.log('[App] parsed payload:', payload);

  // Now it's safe to return early - all hooks have been called
  if (!payload) {
    return (
      <div style={{
        padding: 16,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: '300px'
      }}>
        <style>{`
          @keyframes pulse-glow {
            0%, 100% {
              opacity: 1;
              transform: scale(1);
            }
            50% {
              opacity: 0.4;
              transform: scale(0.95);
            }
          }
        `}</style>
        <svg
          width="80"
          height="80"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
          style={{
            animation: 'pulse-glow 1.2s ease-in-out infinite',
            marginBottom: '20px',
            color: '#f59e0b'
          }}
        >
          <path d="M9 18h6"></path>
          <path d="M10 22h4"></path>
          <path d="M15.09 14c.18-.98.65-1.74 1.41-2.5A4.65 4.65 0 0 0 18 8 6 6 0 0 0 6 8c0 1 .23 2.23 1.5 3.5A4.61 4.61 0 0 1 8.91 14"></path>
          <line x1="9" y1="9" x2="9.01" y2="9"></line>
          <line x1="15" y1="9" x2="15.01" y2="9"></line>
        </svg>
        <p style={{
          fontSize: 16,
          color: '#374151',
          margin: 0,
          fontWeight: 500
        }}>
          Mining product insights from social media chatter ...
        </p>
      </div>
    );
  }

  const { overview, business_insights, key_insights_summary, posts } = payload;

  // Helper to convert SummaryBullet arrays from key_insights_summary to InsightItem[] with equal distribution
  const summaryToInsights = (bullets: SummaryBullet[]): InsightItem[] => {
    if (!bullets || bullets.length === 0) return [];
    const equalPercentage = 100 / bullets.length;
    return bullets.map(bullet => ({
      theme: bullet.text,
      count: bullet.source_indices.length,
      percentage: equalPercentage,
      source_indices: bullet.source_indices
    }));
  };

  // Convert InsightItem[] to ThemeItem[] - backend now provides count and percentage
  const themeItems = (items: InsightItem[], category: Category): ThemeItem[] =>
    items.map((item) => ({
      category,
      theme: item.theme,
      count: item.count,
      pct: item.percentage / 100,  // Convert percentage (0-100) to decimal (0-1)
      source_indices: item.source_indices
    }));

  // Use key_insights_summary if available, otherwise fallback to business_insights
  const likedInsights = key_insights_summary?.what_users_like
    ? summaryToInsights(key_insights_summary.what_users_like)
    : business_insights.what_users_like;

  const dislikedInsights = key_insights_summary?.major_frustrations
    ? summaryToInsights(key_insights_summary.major_frustrations)
    : business_insights.what_users_dont_like;

  const wishesInsights = key_insights_summary?.what_users_want
    ? summaryToInsights(key_insights_summary.what_users_want)
    : business_insights.what_users_wish_existed;

  const liked = themeItems(likedInsights, "like");
  const disliked = themeItems(dislikedInsights, "dislike");
  const wishes = themeItems(wishesInsights, "wish");

  return (
    <div style={{ fontFamily: "system-ui, sans-serif", padding: 16 }}>
      <div style={{
        borderLeft: "4px solid #2563eb",
        background: "#eff6ff",
        color: "#1e3a8a",
        padding: 12,
        borderRadius: 4,
        marginBottom: 16
      }}>
        <h2 style={{ margin: 0, fontWeight: 700 }}>{overview.query.charAt(0).toUpperCase() + overview.query.slice(1)} Product Insights</h2>
      </div>

      <section style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 16 }}>
        <ThemeColumn title="What Users Like" items={liked} tone="positive" onSelect={setDrawer} />
        <ThemeColumn title="What Users Don't Like" items={disliked} tone="warning" onSelect={setDrawer} />
        <ThemeColumn title="What Users Wish Existed" items={wishes} tone="info" onSelect={setDrawer} />
      </section>

      {drawer.open && drawer.theme && drawer.category && drawer.source_indices && (
        <Drawer
          title={`Sources • ${drawer.theme}`}
          onClose={() => setDrawer({ open: false })}
        >
          <ul style={{ margin: 0, padding: 0, listStyle: "none" }}>
            {drawer.source_indices.map((postIndex, i) => {
              const post = payload.posts_with_urls[postIndex];
              if (!post) return null;
              return (
              <li key={i} style={{ padding: "12px 0", borderTop: i === 0 ? "none" : "1px solid #eee" }}>
                {post.url === "URL unavailable" ? (
                  <span style={{ color: "#6b7280", fontSize: 14 }}>{post.title}</span>
                ) : (
                  <a
                    href={post.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    style={{
                      color: "#2563eb",
                      textDecoration: "none",
                      fontSize: 14,
                      display: "block",
                      lineHeight: 1.5
                    }}
                  >
                    {post.title}
                  </a>
                )}
              </li>
              );
            })}
          </ul>
        </Drawer>
      )}
    </div>
  );
}

function ThemeColumn({
  title,
  items,
  tone,
  onSelect,
}: {
  title: string;
  items: ThemeItem[];
  tone: "positive" | "warning" | "info";
  onSelect: (next: { theme: string; category: Category; open: boolean; source_indices?: number[] }) => void;
}) {
  return (
    <div>
      <h3 style={{ marginBottom: 8 }}>{title}</h3>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
        {items.map((t) => (
          <Chip
            key={`${t.category}:${t.theme}`}
            label={t.theme}
            tone={tone}
            disabled={!t.source_indices || t.source_indices.length === 0}
            onClick={() => onSelect({ theme: t.theme, category: t.category, open: true, source_indices: t.source_indices })}
          />
        ))}
      </div>
    </div>
  );
}

function Chip({ label, tone, onClick, disabled }: { label: string; tone: "positive" | "warning" | "info"; onClick: () => void; disabled?: boolean }) {
  const toneColor = tone === "positive" ? "#10b981" : tone === "warning" ? "#f59e0b" : "#3b82f6";

  // Parse "Heading: Context" format
  const colonIndex = label.indexOf(':');
  const heading = colonIndex > 0 ? label.substring(0, colonIndex).trim() : label;
  const context = colonIndex > 0 ? label.substring(colonIndex + 1).trim() : '';

  return (
    <button
      onClick={disabled ? undefined : onClick}
      disabled={disabled}
      style={{
        border: `1px solid ${disabled ? "#d1d5db" : toneColor}`,
        color: disabled ? "#9ca3af" : toneColor,
        background: "white",
        borderRadius: 6,
        padding: "16px 20px",
        display: "block",
        cursor: disabled ? "not-allowed" : "pointer",
        textAlign: "left",
        maxWidth: "100%",
        lineHeight: 1.5,
        opacity: disabled ? 0.6 : 1,
      }}
    >
      <span style={{ fontWeight: 700, color: toneColor }}>
        {heading}
        {context && ": "}
      </span>
      {context && (
        <span style={{ fontSize: 14, color: "#374151" }}>
          {context}
        </span>
      )}
    </button>
  );
}

function Drawer({ title, onClose, children }: { title: string; onClose: () => void; children: React.ReactNode }) {
  return (
    <div
      role="dialog"
      aria-modal="true"
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(0,0,0,0.4)",
        display: "flex",
        justifyContent: "flex-end",
      }}
      onClick={onClose}
    >
      <div
        style={{
          width: "min(720px, 80vw)",
          height: "100%",
          background: "white",
          padding: 16,
          overflow: "auto",
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <h3>{title}</h3>
          <button onClick={onClose} style={{ border: 0, background: "transparent", fontSize: 18, cursor: "pointer" }}>
            ×
          </button>
        </div>
        <div>{children}</div>
      </div>
    </div>
  );
}

// Bootstrap when loaded in the iframe
const rootEl = document.getElementById("root");
if (rootEl) {
  const root = createRoot(rootEl);
  root.render(<App />);
}
