/**
 * API client for the Flask backend.
 */

import { getMockPrediction } from "./mockPredictor";

export interface PredictionResult {
  label: string;
  severity_index: number;
  confidence: number;
  probabilities: Record<string, number>;
  all_probabilities?: Record<string, number>;
  explanation: string;
  heatmap: string;
  overlay?: string;
  preprocessing: string[];
  is_mock: boolean;
  heatmap_is_mock?: boolean;
  needs_doctor: boolean;
  gradcam_layer?: string | null;
  gradcam_method?: string | null;
}

const API_BASE_URL = "http://localhost:5001";

function getMaxProbability(
  probabilities?: Record<string, number>,
  fallback = 0,
): number {
  if (!probabilities) {
    return fallback;
  }

  const values = Object.values(probabilities).filter(
    (value) => typeof value === "number" && Number.isFinite(value),
  );

  if (values.length === 0) {
    return fallback;
  }

  return Math.max(...values);
}

function normalizePredictionResult(data: PredictionResult): PredictionResult {
  const probabilities = data.probabilities ?? data.all_probabilities ?? {};
  const confidence = getMaxProbability(probabilities, data.confidence ?? 0);

  return {
    ...data,
    probabilities,
    all_probabilities: data.all_probabilities ?? probabilities,
    confidence,
    overlay: data.overlay ?? data.heatmap,
  };
}

export async function checkHealth(): Promise<{
  online: boolean;
  mode: string;
}> {
  try {
    const response = await fetch(`${API_BASE_URL}/health`, {
      signal: AbortSignal.timeout(3000),
    });

    if (response.ok) {
      const data = await response.json();
      return { online: true, mode: data.mode || "LIVE" };
    }

    return { online: false, mode: "MOCK" };
  } catch {
    return { online: false, mode: "MOCK" };
  }
}

export async function predictDR(file: File): Promise<PredictionResult> {
  const formData = new FormData();
  formData.append("image", file);

  try {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: "POST",
      body: formData,
      signal: AbortSignal.timeout(30000),
    });

    if (!response.ok) {
      const errorData = await response
        .json()
        .catch(() => ({ error: "Unknown server error" }));
      throw new Error(errorData.error || `Server error: ${response.status}`);
    }

    const data = (await response.json()) as PredictionResult;
    return normalizePredictionResult(data);
  } catch (error) {
    if (error instanceof TypeError && error.message.includes("fetch")) {
      console.warn("Backend unreachable; using mock predictions");
      return normalizePredictionResult(await getMockPrediction(file));
    }

    if (error instanceof DOMException && error.name === "AbortError") {
      console.warn("Backend timed out; using mock predictions");
      return normalizePredictionResult(await getMockPrediction(file));
    }

    throw error;
  }
}
