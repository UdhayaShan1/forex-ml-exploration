import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  base: "/forex-ml-exploration/",
  build: {
    outDir: "dist"
  }
});
