import { useEffect, useState } from "react";
import { Outlet } from "react-router-dom";
import { HardHat } from "lucide-react";
import Sidebar from "./components/Sidebar";
import { api } from "./lib/api";
import type { HealthStatus } from "./lib/types";

export default function App() {
  const [health, setHealth] = useState<HealthStatus | null>(null);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null));
    const t = setInterval(() => api.health().then(setHealth).catch(() => {}), 30_000);
    return () => clearInterval(t);
  }, []);

  return (
    <div className="flex h-full min-h-0">
      <Sidebar health={health} />
      <main className="relative flex-1 min-w-0 overflow-hidden">
        <div
          aria-hidden
          className="pointer-events-none absolute inset-0 opacity-[0.5]"
          style={{
            background:
              "radial-gradient(900px 400px at 75% -10%, rgba(245,165,36,0.07), transparent 60%), radial-gradient(700px 500px at 10% 110%, rgba(53,208,186,0.05), transparent 60%)",
          }}
        />
        <div className="relative h-full min-h-0 overflow-y-auto">
          <Outlet />
        </div>
      </main>
    </div>
  );
}

export function Logo({ size = 34 }: { size?: number }) {
  return (
    <div
      className="grid place-items-center rounded-xl shadow-lg shadow-amber-500/10"
      style={{
        width: size,
        height: size,
        background: "linear-gradient(135deg, #f5a524, #fb7822)",
      }}
    >
      <HardHat size={size * 0.56} strokeWidth={2.4} className="text-ink-950" />
    </div>
  );
}
