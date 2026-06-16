export const API_BASE =
  import.meta.env.VITE_API_BASE_URL ??
  `${window.location.protocol}//${window.location.hostname}:8000`;

export const CLASS_LABELS = [
  "adulterated",
  "bark_average",
  "bark_ordinary",
  "bark_special",
  "bark_superior",
  "leaf_fail",
  "leaf_pass",
] as const;

export type QualityTier = "green" | "amber" | "red";

export const QUALITY_MAP: Record<string, QualityTier> = {
  bark_superior: "green",
  bark_special: "green",
  bark_average: "amber",
  bark_ordinary: "amber",
  leaf_pass: "amber",
  leaf_fail: "red",
  adulterated: "red",
};

export const QUALITY_LABEL: Record<QualityTier, string> = {
  green: "High Quality",
  amber: "Moderate",
  red: "Low Quality",
};
