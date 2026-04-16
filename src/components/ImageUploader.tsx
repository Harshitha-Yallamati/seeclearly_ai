import { useCallback, useState } from "react";
import { Upload, X, ImageIcon } from "lucide-react";

interface ImageUploaderProps {
  onImageSelected: (file: File, preview: string) => void;
  onClear: () => void;
  preview: string | null;
}

const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB

const ImageUploader = ({ onImageSelected, onClear, preview }: ImageUploaderProps) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const validateAndProcess = useCallback(
    (file: File) => {
      setError(null);

      if (!file.type.startsWith("image/")) {
        setError("Please upload an image file (PNG, JPG, BMP, etc.)");
        return;
      }

      if (file.size > MAX_FILE_SIZE) {
        setError(`File too large (${(file.size / 1024 / 1024).toFixed(1)}MB). Maximum: 10MB.`);
        return;
      }

      const reader = new FileReader();
      reader.onload = (e) => {
        onImageSelected(file, e.target?.result as string);
      };
      reader.readAsDataURL(file);
    },
    [onImageSelected]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file) validateAndProcess(file);
    },
    [validateAndProcess]
  );

  // Preview mode
  if (preview) {
    return (
      <div className="relative rounded-xl overflow-hidden border border-border/50 glass group">
        <img
          src={preview}
          alt="Uploaded fundus image"
          className="w-full h-72 object-contain bg-black/10"
        />
        <button
          onClick={onClear}
          className="absolute top-3 right-3 p-2 rounded-full bg-black/60 backdrop-blur-sm text-white hover:bg-destructive transition-colors"
          title="Remove image"
        >
          <X className="w-4 h-4" />
        </button>
        <div className="absolute bottom-0 inset-x-0 py-2 px-3 bg-gradient-to-t from-black/50 to-transparent">
          <div className="flex items-center gap-2">
            <ImageIcon className="w-3 h-3 text-white/70" />
            <span className="text-[10px] font-medium text-white/70">Ready for analysis</span>
          </div>
        </div>
      </div>
    );
  }

  // Upload zone
  return (
    <div>
      <label
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragOver(true);
        }}
        onDragLeave={() => setIsDragOver(false)}
        onDrop={handleDrop}
        className={`flex flex-col items-center justify-center h-72 rounded-xl border-2 border-dashed cursor-pointer transition-all duration-300 ${
          isDragOver
            ? "border-primary bg-primary/5 glass-glow scale-[1.01]"
            : "border-border/50 bg-card/30 hover:border-primary/50 hover:bg-card/50"
        }`}
      >
        <div className={`p-4 rounded-2xl mb-4 transition-all duration-300 ${
          isDragOver ? "bg-primary/20 scale-110" : "bg-muted/30"
        }`}>
          <Upload className={`w-8 h-8 transition-colors ${isDragOver ? "text-primary" : "text-muted-foreground"}`} />
        </div>

        <p className="text-sm font-semibold text-foreground">
          {isDragOver ? "Drop it here!" : "Drop a retinal fundus image"}
        </p>
        <p className="text-xs text-muted-foreground mt-1.5">
          or <span className="text-primary font-medium cursor-pointer hover:underline">click to browse</span>
        </p>
        <p className="text-[10px] text-muted-foreground/60 mt-3">
          PNG, JPG, BMP, TIFF • Max 10MB
        </p>

        <input
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) validateAndProcess(file);
          }}
        />
      </label>

      {/* Error message */}
      {error && (
        <div className="mt-2 px-3 py-2 rounded-lg bg-destructive/10 border border-destructive/20">
          <p className="text-xs text-destructive font-medium">{error}</p>
        </div>
      )}
    </div>
  );
};

export default ImageUploader;
