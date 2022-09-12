import { defineConfig } from 'rollup';
import json from '@rollup/plugin-json';

export default defineConfig({
  input: "./dist/examples/mnist/demo.js",
  output: {
    file: "./out/bundle.js",
    format: "iife",
    inlineDynamicImports: true,
  },
  plugins: [
    json()
  ]
});