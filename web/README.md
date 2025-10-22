Custom UX Component (Apps SDK)

Overview
- Built per the Apps SDK “Build a custom UX” guide.
- React component bundled to a single ESM file you can inline in your MCP server responses.

Files
- web/src/component.tsx — main UI (overview cards, three theme boards, evidence drawer)
- web/src/sdk-hooks.ts — lightweight hooks for window.openai bridge (tool outputs, layout, widget state)
- web/src/types.ts — payload types for your analysis JSON
- web/src/derive.ts — keyword matching + snippet extraction
- web/package.json — esbuild script
- web/tsconfig.json — TS config

Build
1) cd web
2) npm install
3) npm run build

This generates dist/component.js

How it receives data
- The component reads the latest tool output via the Apps SDK iframe bridge (window.openai.getToolOutputs / subscribeToolOutputs).
- It also supports local dev fallbacks: set window.__APPS_TOOL_OUTPUTS__ to your payload before mounting.

Embedding in your server
- Follow the “Set up your server” Apps SDK docs to return a component UI template that inlines dist/component.js.
- Provide your payload as the tool result so the iframe bridge exposes it to the component.

Payload shape
- Matches your current analysis JSON: overview, posts[], business_insights{ like|dont_like|wish }, recommendation.

