import { useState } from "react";

interface HeatmapViewerProps {
  originalPreview: string;
  heatmapBase64: string;
}

type ViewMode = "original" | "heatmap" | "overlay";

const HeatmapViewer = ({ originalPreview, heatmapBase64 }: HeatmapViewerProps) => {
  const [mode, setMode] = useState<ViewMode>("overlay");
  const [isZoomed, setIsZoomed] = useState(false);

  const modes: { key: ViewMode; label: string; icon: string }[] = [
    { key: "original", label: "Original", icon: "🔬" },
    { key: "heatmap", label: "Grad-CAM", icon: "🔥" },
    { key: "overlay", label: "Overlay", icon: "🧬" },
  ];

  const getDisplayImage = () => {
    switch (mode) {
      case "original":
        return originalPreview;
      case "heatmap":
      case "overlay":
        return heatmapBase64;
      default:
        return heatmapBase64;
    }
  };

  return (
    <div className="space-y-3 animate-fade-in-up">
      {/* Mode Selector */}
      <div className="flex gap-1.5 p-1 rounded-lg bg-muted/50">
        {modes.map((m) => (
          <button
            key={m.key}
            onClick={() => setMode(m.key)}
            className={`flex-1 flex items-center justify-center gap-1.5 py-2 px-3 rounded-md text-xs font-semibold transition-all duration-200 ${
              mode === m.key
                ? "bg-primary text-primary-foreground shadow-md"
                : "text-muted-foreground hover:text-foreground hover:bg-muted"
            }`}
          >
            <span>{m.icon}</span>
            <span>{m.label}</span>
          </button>
        ))}
      </div>

      {/* Image Display */}
      <div
        className={`relative rounded-xl overflow-hidden border border-border/50 glass cursor-pointer transition-all duration-300 ${
          isZoomed ? "scale-110 z-50 shadow-2xl" : ""
        }`}
        onClick={() => setIsZoomed(!isZoomed)}
      >
        <img
          src={getDisplayImage()}
          alt={`${mode} view of retinal fundus`}
          className="w-full h-64 object-contain bg-black/20"
        />

        {/* Overlay label */}
        <div className="absolute bottom-2 left-2 px-2 py-1 rounded-md bg-black/60 backdrop-blur-sm">
          <span className="text-[10px] font-bold text-white uppercase tracking-wider">
            {mode === "original" ? "Original Fundus" : mode === "heatmap" ? "Grad-CAM Heatmap" : "Model Attention Overlay"}
          </span>
        </div>

        {/* Zoom hint */}
        <div className="absolute bottom-2 right-2 px-2 py-1 rounded-md bg-black/40 backdrop-blur-sm">
          <span className="text-[10px] text-white/60">
            {isZoomed ? "Click to zoom out" : "Click to zoom"}
          </span>
        </div>
      </div>

      {/* Explanation */}
      <p className="text-xs text-muted-foreground leading-relaxed">
        {mode === "original" && "The original retinal fundus image as uploaded."}
        {mode === "heatmap" && "Grad-CAM highlights regions the model focused on. Red/yellow areas indicate high model attention — likely pathological features."}
        {mode === "overlay" && "Overlay shows the model's attention regions mapped onto the original image. Bright areas correlate with detected retinopathy features."}
      </p>
    </div>
  );
};

export default HeatmapViewer;
