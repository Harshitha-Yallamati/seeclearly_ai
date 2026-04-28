import { useState } from "react";

interface HeatmapViewerProps {
  originalPreview: string;
  heatmapBase64: string;
  overlayBase64?: string;
}

type ViewMode = "original" | "heatmap" | "overlay";

const HeatmapViewer = ({
  originalPreview,
  heatmapBase64,
  overlayBase64,
}: HeatmapViewerProps) => {
  const [mode, setMode] = useState<ViewMode>("overlay");
  const [isZoomed, setIsZoomed] = useState(false);

  const modes: { key: ViewMode; label: string; icon: string }[] = [
    { key: "original", label: "Original", icon: "O" },
    { key: "heatmap", label: "Grad-CAM", icon: "H" },
    { key: "overlay", label: "Overlay", icon: "M" },
  ];

  const getDisplayImage = () => {
    switch (mode) {
      case "original":
        return originalPreview;
      case "heatmap":
        return heatmapBase64;
      case "overlay":
      default:
        return overlayBase64 || heatmapBase64;
    }
  };

  return (
    <div className="animate-fade-in-up space-y-3">
      <div className="flex gap-1.5 rounded-lg bg-muted/50 p-1">
        {modes.map((view) => (
          <button
            key={view.key}
            onClick={() => setMode(view.key)}
            className={`flex flex-1 items-center justify-center gap-1.5 rounded-md px-3 py-2 text-xs font-semibold transition-all duration-200 ${
              mode === view.key
                ? "bg-primary text-primary-foreground shadow-md"
                : "text-muted-foreground hover:bg-muted hover:text-foreground"
            }`}
          >
            <span className="font-mono text-[11px]">{view.icon}</span>
            <span>{view.label}</span>
          </button>
        ))}
      </div>

      <div
        className={`relative cursor-pointer overflow-hidden rounded-xl border border-border/50 glass transition-all duration-300 ${
          isZoomed ? "z-50 scale-110 shadow-2xl" : ""
        }`}
        onClick={() => setIsZoomed((value) => !value)}
      >
        <img
          src={getDisplayImage()}
          alt={`${mode} view of retinal fundus`}
          className="h-64 w-full bg-black/20 object-contain"
        />

        <div className="absolute bottom-2 left-2 rounded-md bg-black/60 px-2 py-1 backdrop-blur-sm">
          <span className="text-[10px] font-bold uppercase tracking-wider text-white">
            {mode === "original"
              ? "Original Fundus"
              : mode === "heatmap"
                ? "Raw JET Heatmap"
                : "Alpha-Blended Overlay"}
          </span>
        </div>

        <div className="absolute bottom-2 right-2 rounded-md bg-black/40 px-2 py-1 backdrop-blur-sm">
          <span className="text-[10px] text-white/60">
            {isZoomed ? "Click to zoom out" : "Click to zoom"}
          </span>
        </div>
      </div>

      <p className="text-xs leading-relaxed text-muted-foreground">
        {mode === "original" &&
          "The original retinal fundus image as uploaded."}
        {mode === "heatmap" &&
          "The raw Grad-CAM heatmap shows where the network assigned the strongest activation. Warmer colors indicate higher attention."}
        {mode === "overlay" &&
          "The overlay blends the normalized JET heatmap onto the original retina so lesion-focused regions stay aligned with the anatomy."}
      </p>
    </div>
  );
};

export default HeatmapViewer;
