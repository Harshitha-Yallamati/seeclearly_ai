/**
 * SeeClearly AI — API Client
 *
 * Handles communication with the Flask backend.
 * Falls back to mock predictions when the backend is unreachable.
 */

import { getMockPrediction } from "./mockPredictor";

export interface PredictionResult {
  label: string;
  severity_index: number;
  confidence: number;
  all_probabilities?: Record<string, number>;
  explanation: string;
  heatmap: string;          // base64 data URI
  preprocessing: string[];
  is_mock: boolean;
  needs_doctor: boolean;
}

const API_BASE_URL = "http://localhost:5001";

/**
 * Check if the Flask backend is online.
 */
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

/**
 * Send an image to the backend for DR prediction.
 * Falls back to mock predictions if the backend is unreachable.
 */
export async function predictDR(file: File): Promise<PredictionResult> {
  const formData = new FormData();
  formData.append("image", file);

  try {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: "POST",
      body: formData,
      signal: AbortSignal.timeout(30000), // 30s timeout for model inference
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: "Unknown error" }));
      throw new Error(errorData.error || `Server error: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    // If it's a network error (backend down), use mock
    if (error instanceof TypeError && error.message.includes("fetch")) {
      console.warn("Backend unreachable — using mock predictions");
      return getMockPrediction(file);
    }

    // If it's an AbortError (timeout), use mock
    if (error instanceof DOMException && error.name === "AbortError") {
      console.warn("Backend timeout — using mock predictions");
      return getMockPrediction(file);
    }

    // Re-throw actual server errors
    throw error;
  }
}
