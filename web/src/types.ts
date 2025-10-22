export type InsightItem = {
  theme: string;
  count: number;
  percentage: number;
  source_indices?: number[];
};

export type PostSource = {
  title: string;
  url: string;
};

export type SummaryBullet = {
  text: string;
  source_indices: number[];
};

export type Payload = {
  overview: {
    query: string;
    total_posts: number;
    subreddits_searched: string[];
    analysis_method: string;
  };
  posts: string[];
  posts_with_urls: PostSource[];
  business_insights: {
    what_users_like: InsightItem[];
    what_users_dont_like: InsightItem[];
    what_users_wish_existed: InsightItem[];
  };
  key_insights_summary?: {
    what_users_like: SummaryBullet[];
    major_frustrations: SummaryBullet[];
    what_users_want: SummaryBullet[];
  };
  recommendation: string;
};

export type Category = "like" | "dislike" | "wish";

export type ThemeItem = {
  category: Category;
  theme: string;
  count: number;
  pct: number; // 0..1
  source_indices?: number[];
};

export type Match = {
  index: number;
  title: string;
  snippet: string;
};

