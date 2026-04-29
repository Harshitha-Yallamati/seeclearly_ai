import { useState } from "react";
import { Eye, Loader2, RotateCcw, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import ImageUploader from "@/components/ImageUploader";
import ResultDisplay from "@/components/ResultDisplay";
import AnalysisAnimation from "@/components/AnalysisAnimation";
import ConnectionStatus from "@/components/ConnectionStatus";
import { predictDR, type PredictionResult } from "@/lib/api";

const Index = () => {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleImageSelected = (f: File, p: string) => {
    setFile(f);
    setPreview(p);
    setResult(null);
    setError(null);
  };

  const handleClear = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setIsAnalyzing(true);
    setError(null);
    try {
      const prediction = await predictDR(file);
      setResult(prediction);
    } catch (err) {
      console.error("Analysis failed:", err);
      setError(err instanceof Error ? err.message : "Analysis failed. Please try again.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Background gradient */}
      <div className="fixed inset-0 pointer-events-none">
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-primary/3 rounded-full blur-3xl" />
        <div className="absolute bottom-1/4 right-1/4 w-80 h-80 bg-primary/2 rounded-full blur-3xl" />
      </div>

      {/* Header */}
      <header className="sticky top-0 z-50 glass-strong border-b border-border/30">
        <div className="max-w-2xl mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex items-center justify-center">
              <img src="/placeholder.svg" alt="Logo" className="w-8 h-8 object-contain" />
            </div>
            <div>
              <h1 className="text-lg font-extrabold text-foreground tracking-tight leading-none">
                Retino<span className="text-primary">Check</span>
              </h1>
              <p className="text-[9px] font-semibold text-muted-foreground uppercase tracking-[0.2em] mt-0.5">
                Diabetic Retinopathy Detection
              </p>
            </div>
          </div>
          <ConnectionStatus />
        </div>
      </header>

      {/* Main Content */}
      <main className="relative max-w-2xl mx-auto px-4 py-8 space-y-6">
        {/* Hero — only show when no file is selected */}
        {!file && (
          <div className="text-center space-y-3 py-4 animate-fade-in-up">
            <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-primary/10 border border-primary/20 text-primary text-xs font-semibold">
              <Sparkles className="w-3 h-3" />
              AI-Powered Retinal Analysis
            </div>
            <h2 className="text-2xl font-black text-foreground tracking-tight">
              Early Detection Saves Vision
            </h2>
            <p className="text-sm text-muted-foreground max-w-md mx-auto leading-relaxed">
              Upload a retinal fundus image for instant AI-powered Diabetic Retinopathy screening with Grad-CAM explainability.
            </p>
          </div>
        )}

        {/* Step 1: Upload */}
        <section className="animate-fade-in-up">
          <div className="flex items-center gap-2 mb-3">
            <div className="w-6 h-6 rounded-full bg-primary/20 flex items-center justify-center">
              <span className="text-xs font-bold text-primary">1</span>
            </div>
            <h2 className="text-sm font-bold text-foreground">
              Upload Fundus Image
            </h2>
          </div>
          <ImageUploader
            onImageSelected={handleImageSelected}
            onClear={handleClear}
            preview={!isAnalyzing ? preview : null}
          />
        </section>

        {/* Analysis Animation (while processing) */}
        {isAnalyzing && preview && (
          <AnalysisAnimation imagePreview={preview} />
        )}

        {/* Analyze button */}
        {file && !result && !isAnalyzing && (
          <Button
            id="analyze-button"
            onClick={handleAnalyze}
            className="w-full h-12 text-base font-bold bg-gradient-to-r from-primary to-primary/80 hover:from-primary/90 hover:to-primary/70 shadow-lg shadow-primary/20 transition-all duration-300 hover:shadow-primary/30 hover:scale-[1.01]"
          >
            <Eye className="w-5 h-5 mr-2" />
            Analyze Image
          </Button>
        )}

        {/* Error message */}
        {error && (
          <div className="flex items-start gap-3 p-4 rounded-xl bg-destructive/10 border border-destructive/20 animate-fade-in-up">
            <span className="text-destructive text-lg">⚠️</span>
            <div>
              <p className="text-sm font-semibold text-destructive">Analysis Failed</p>
              <p className="text-xs text-destructive/80 mt-1">{error}</p>
            </div>
            <Button
              variant="ghost"
              size="sm"
              onClick={handleAnalyze}
              className="ml-auto text-xs"
            >
              <RotateCcw className="w-3 h-3 mr-1" />
              Retry
            </Button>
          </div>
        )}

        {/* Step 2: Results */}
        {result && preview && (
          <section>
            <div className="flex items-center gap-2 mb-3">
              <div className="w-6 h-6 rounded-full bg-primary/20 flex items-center justify-center">
                <span className="text-xs font-bold text-primary">2</span>
              </div>
              <h2 className="text-sm font-bold text-foreground">
                Analysis Results
              </h2>
            </div>
            <ResultDisplay result={result} originalPreview={preview} />

            {/* New Analysis button */}
            <div className="mt-6">
              <Button
                id="new-analysis-button"
                variant="outline"
                onClick={handleClear}
                className="w-full h-10 text-sm font-semibold"
              >
                <RotateCcw className="w-4 h-4 mr-2" />
                New Analysis
              </Button>
            </div>
          </section>
        )}
      </main>

      {/* Footer */}
      <footer className="max-w-2xl mx-auto px-4 py-6 border-t border-border/20">
        <p className="text-[10px] text-muted-foreground/40 text-center">
          RetinoCheck © {new Date().getFullYear()} — For research and educational purposes only.
        </p>
      </footer>
    </div>
  );
};

export default Index;
