import { useEffect, useState } from "react";

const API_BASE_URL = "http://localhost:5001";

type Status = "checking" | "connected" | "disconnected";

const ConnectionStatus = () => {
  const [status, setStatus] = useState<Status>("checking");
  const [mode, setMode] = useState<string>("");

  const checkHealth = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/health`, {
        signal: AbortSignal.timeout(3000),
      });
      if (response.ok) {
        const data = await response.json();
        setStatus("connected");
        setMode(data.mode || "LIVE");
      } else {
        setStatus("disconnected");
        setMode("MOCK");
      }
    } catch {
      setStatus("disconnected");
      setMode("MOCK");
    }
  };

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 15000); // Check every 15s
    return () => clearInterval(interval);
  }, []);

  const config = {
    checking:     { color: "bg-yellow-400", text: "Checking...",  textColor: "text-yellow-400" },
    connected:    { color: "bg-emerald-400", text: "Backend Online", textColor: "text-emerald-400" },
    disconnected: { color: "bg-orange-400",  text: "Mock Mode",     textColor: "text-orange-400" },
  };

  const c = config[status];

  return (
    <div className="flex items-center gap-2 px-2.5 py-1 rounded-full bg-muted/50" title={mode || status}>
      <div className={`w-2 h-2 rounded-full ${c.color} ${status === "connected" ? "animate-pulse" : ""}`} />
      <span className={`text-[10px] font-semibold uppercase tracking-wider ${c.textColor}`}>
        {c.text}
      </span>
    </div>
  );
};

export default ConnectionStatus;
