/* Lightweight hooks mirroring the patterns shown in the Apps SDK Custom UX guide.
 * These wrap the iframe bridge (window.openai) and provide React ergonomics.
 * Uses the official ChatGPT Apps SDK API with useSyncExternalStore pattern.
 * Reference: https://developers.openai.com/apps-sdk/build/custom-ux
 */
import { useCallback, useSyncExternalStore, useState } from "react";

type AnyJSON = any;

const SET_GLOBALS_EVENT_TYPE = "openai:set_globals";

type OpenAiGlobals = {
  toolOutput?: AnyJSON;
  toolInput?: AnyJSON;
  toolResponseMetadata?: AnyJSON;
  theme?: string;
  userAgent?: AnyJSON;
  locale?: string;
  layout?: AnyJSON;
  displayMode?: string;
  widgetState?: AnyJSON;
};

declare global {
  interface Window {
    openai?: OpenAiGlobals & {
      // Methods
      setWidgetState?: (next: AnyJSON) => void;
      callTool?: (name: string, args: AnyJSON) => Promise<AnyJSON>;
      requestDisplayMode?: (opts: { mode: string }) => Promise<void>;
    };

    // Fallbacks for local dev
    __APPS_TOOL_OUTPUTS__?: AnyJSON;
    __APPS_LAYOUT__?: AnyJSON;
    __APPS_WIDGET_STATE__?: AnyJSON;
  }
}

/**
 * Official pattern from OpenAI Apps SDK docs
 * Subscribes to window.openai.toolOutput using the openai:set_globals event
 */
export function useOpenAiGlobal<K extends keyof OpenAiGlobals>(
  key: K
): OpenAiGlobals[K] {
  return useSyncExternalStore(
    (onChange) => {
      const handleSetGlobal = () => {
        const value = window.openai?.[key];
        if (value !== undefined) {
          onChange();
        }
      };
      window.addEventListener(SET_GLOBALS_EVENT_TYPE, handleSetGlobal, {
        passive: true,
      });
      return () => {
        window.removeEventListener(SET_GLOBALS_EVENT_TYPE, handleSetGlobal);
      };
    },
    () => window.openai?.[key]
  );
}

/**
 * Hook to access tool output data from ChatGPT
 * Uses the official useSyncExternalStore pattern
 */
export function useToolOutputs<T = AnyJSON>(initial?: T): T | undefined {
  const toolOutput = useOpenAiGlobal('toolOutput');

  console.log('[useToolOutputs] toolOutput from useOpenAiGlobal:', toolOutput);

  // Fallback for local dev
  if (toolOutput === undefined && window.__APPS_TOOL_OUTPUTS__) {
    return window.__APPS_TOOL_OUTPUTS__ as T;
  }

  return (toolOutput as T) ?? initial;
}

/**
 * Hook to access layout globals (maxHeight, displayMode, etc.)
 * Uses the official useSyncExternalStore pattern
 */
export function useLayout<T = AnyJSON>(initial?: T): T | undefined {
  const layout = useOpenAiGlobal('layout');

  // Fallback for local dev
  if (layout === undefined && window.__APPS_LAYOUT__) {
    return window.__APPS_LAYOUT__ as T;
  }

  return (layout as T) ?? initial;
}

/**
 * Hook to manage widget state that persists in ChatGPT
 * Uses the official useSyncExternalStore pattern for reading
 * and window.openai.setWidgetState for writing
 */
export function useWidgetState<T = AnyJSON>(initial: T): [T, (updater: T | ((prev: T) => T)) => void] {
  // Subscribe to widgetState changes using the official pattern
  const widgetState = useOpenAiGlobal('widgetState');

  // Use local state with sync to window.openai
  const [state, setState] = useState<T>(() => {
    const fromSdk = widgetState ?? window.openai?.widgetState;
    return (fromSdk as T) ?? (window.__APPS_WIDGET_STATE__ as T) ?? initial;
  });

  const set = useCallback(
    (updater: T | ((prev: T) => T)) => {
      setState((prev: T) => {
        const nextState = typeof updater === "function" ? (updater as (p: T) => T)(prev) : updater;
        // Persist state to ChatGPT host
        window.openai?.setWidgetState?.(nextState);
        return nextState;
      });
    },
    []
  );

  return [state, set] as const;
}
