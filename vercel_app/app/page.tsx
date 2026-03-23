"use client";

import { useChat } from "ai/react";
import { useEffect, useRef } from "react";

export default function Page() {
  const { messages, input, handleInputChange, handleSubmit, status } = useChat({
    api: "/api/chat",
  });

  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const isLoading = status === "submitted" || status === "streaming";

  function renderContent(content: string) {
    return content.split("\n").map((line, i) => {
      if (line.startsWith("## ") || line.startsWith("### ") || line.startsWith("#### ")) {
        return (
          <h2 key={i} className="text-sm font-semibold text-white mt-4 mb-1">
            {line.replace(/^#{2,4} /, "")}
          </h2>
        );
      }
      if (line.startsWith("- ") || line.startsWith("· ")) {
        return (
          <p key={i} className="text-sm text-neutral-300 ml-2 mb-0.5">
            · {line.replace(/^[-·] /, "").replace(/\*\*/g, "")}
          </p>
        );
      }
      if (line.trim()) {
        return (
          <p key={i} className="text-sm text-neutral-300 mb-1">
            {line.replace(/\*\*/g, "")}
          </p>
        );
      }
      return null;
    });
  }

  return (
    <main className="flex flex-col h-screen max-w-2xl mx-auto px-4">
      {/* Header */}
      <div className="py-6 border-b border-neutral-800">
        <h1 className="text-base font-semibold">Customer Intelligence Engine</h1>
        <p className="text-xs text-neutral-500 mt-0.5">
          Vercel AI SDK · Reddit MCP · Streaming
        </p>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto py-6 space-y-6">
        {messages.length === 0 && (
          <div className="text-center text-neutral-600 text-sm mt-16">
            <p>Ask about any product or company.</p>
            <p className="mt-1">Try: "What do people think about Cursor?"</p>
          </div>
        )}

        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
          >
            {msg.role === "user" ? (
              <div className="bg-neutral-800 text-white text-sm rounded-2xl rounded-tr-sm px-4 py-2 max-w-xs">
                {msg.content}
              </div>
            ) : (
              <div className="max-w-prose">
                {msg.toolInvocations?.map((t) => (
                  <div
                    key={t.toolCallId}
                    className="text-xs text-neutral-500 mb-3 flex items-center gap-1.5"
                  >
                    <span className="inline-block w-1.5 h-1.5 rounded-full bg-neutral-500 animate-pulse" />
                    Fetching Reddit data for &ldquo;{t.args?.query}&rdquo;...
                  </div>
                ))}
                {msg.content && (
                  <div className="bg-neutral-900 border border-neutral-800 rounded-2xl rounded-tl-sm px-4 py-3">
                    {renderContent(msg.content)}
                  </div>
                )}
              </div>
            )}
          </div>
        ))}

        {isLoading && messages[messages.length - 1]?.role === "user" && (
          <div className="flex justify-start">
            <div className="flex items-center gap-1.5 px-4 py-3">
              <span className="w-1.5 h-1.5 rounded-full bg-neutral-500 animate-bounce [animation-delay:0ms]" />
              <span className="w-1.5 h-1.5 rounded-full bg-neutral-500 animate-bounce [animation-delay:150ms]" />
              <span className="w-1.5 h-1.5 rounded-full bg-neutral-500 animate-bounce [animation-delay:300ms]" />
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="py-4 border-t border-neutral-800">
        <form onSubmit={handleSubmit} className="flex gap-2">
          <input
            value={input}
            onChange={handleInputChange}
            placeholder="Ask about any product..."
            disabled={isLoading}
            className="flex-1 bg-neutral-900 border border-neutral-700 rounded-xl px-4 py-2.5 text-sm text-white placeholder-neutral-500 focus:outline-none focus:border-neutral-500 disabled:opacity-40"
          />
          <button
            type="submit"
            disabled={isLoading || !input.trim()}
            className="bg-white text-black text-sm font-medium rounded-xl px-4 py-2.5 hover:bg-neutral-200 transition disabled:opacity-40 disabled:cursor-not-allowed"
          >
            Send
          </button>
        </form>
      </div>
    </main>
  );
}
