import { ReLU, Softmax } from "../../src/act";
import { Parameter } from "../../src/autograd";
import { Conv1d, Conv2d } from "../../src/conv";
import { Dropout } from "../../src/dropout";
import { Flatten } from "../../src/flatten";
import { Linear } from "../../src/linear";
import { Sequential } from "../../src/sequential";
import { savedModelParams } from "./trainedModelParams"
const drawCanvas = document.getElementById("draw") as HTMLCanvasElement;
const canvas1 = document.getElementById("display1") as HTMLCanvasElement;
const canvas2 = document.getElementById("display2") as HTMLCanvasElement;
const canvas3 = document.getElementById("display3") as HTMLCanvasElement;
const resetButton = document.getElementById("btn-reset") as HTMLButtonElement;
const guessDiv = document.getElementById("result-guess") as HTMLDivElement;
const pixelMultiplier = 14;
[drawCanvas, canvas1, canvas2, canvas3].forEach(canvas => {
  canvas.width = 28 * pixelMultiplier;
  canvas.height = 28 * pixelMultiplier;
});
const drawCtx = drawCanvas.getContext("2d")!;


resetButton.addEventListener("click", () => {
  [drawCanvas, canvas1, canvas2, canvas3].forEach(canvas => {
    const ctx = canvas.getContext("2d")!;
    ctx.fillStyle = "#000"
    ctx.fillRect(0, 0, canvas.width, canvas.height)
  });
})

drawCtx.fillStyle = "#000"
drawCtx.fillRect(0, 0, drawCanvas.width, drawCanvas.height)
let mousedown = false;

const startDrawing = (x: number, y: number) => {
  mousedown = true
  drawCtx.fillStyle = "#fff"
  drawCtx.strokeStyle = "#fff"
  drawCtx.beginPath();
  drawCtx.moveTo(x, y);
}

const keepDrawing = (x: number, y: number) => {
  if(mousedown) {
    drawCtx.lineWidth = pixelMultiplier * 3;
    drawCtx.lineCap = "round"
    drawCtx.lineJoin = "round"
    drawCtx.lineTo(x, y)
    drawCtx.stroke();
    drawCtx.moveTo(x, y);
    guessWithCooldown();
  }
}
const stopDrawing = () => {
  mousedown = false
  drawCtx.closePath();
  guess();
}

drawCanvas.addEventListener("mousedown", e => startDrawing(e.offsetX, e.offsetY));
drawCanvas.addEventListener("touchstart", e => { e.preventDefault(); startDrawing(e.touches[0].clientX - drawCanvas.offsetLeft, e.touches[0].clientY - drawCanvas.offsetTop) });
drawCanvas.addEventListener("mousemove", e => keepDrawing(e.offsetX, e.offsetY));
drawCanvas.addEventListener("touchmove", e => { e.preventDefault(); keepDrawing(e.touches[0].clientX - drawCanvas.offsetLeft, e.touches[0].clientY - drawCanvas.offsetTop) });
document.body.addEventListener("mouseup", stopDrawing);
document.body.addEventListener("touchup", stopDrawing);

const model = new Sequential(
  new Conv2d(4, 3, 2, 0),
  new ReLU(),
  new Conv2d(4, 3, 2, 0),
  new ReLU(),
  new Flatten(),
  new Linear(144, 10),
  new Softmax(),
);

model.loadFromValues(savedModelParams);

const drawGrayscaleOnCanvas = (canvas: HTMLCanvasElement, img: number[][]) => {
  const ctx = canvas.getContext("2d")!;
  const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const repeat = Math.ceil(canvas.width / img[0].length);
  for(let y = 0; y < img.length; y++) {
    for(let x = 0; x < img[0].length; x++) {
      const val = img[y][x];
      for(let i = 0; i < repeat; i++) {
        for(let j = 0; j < repeat; j++) {
          const idx = (y * repeat + j) * canvas.width * 4 + (x * repeat + i) * 4;
          imgData.data[idx] = val * 255;
          imgData.data[idx + 1] = val * 255;
          imgData.data[idx + 2] = val * 255;
          imgData.data[idx + 3] = 255;
        }
      }
    }
  }
  ctx.putImageData(imgData, 0, 0);

}

const downsizeGrayscale = (img: number[][], shape: [number, number]) => {
  const [width, height] = shape;
  const wStep = Math.ceil(img[0].length / width);
  const hStep = Math.ceil(img.length / height);
  return Array.from({ length: height }, (_, y) =>
    Array.from({ length: width }, (_, x) => {
      let sum = 0;
      for(let i = 0; i < wStep; i++) {
        for(let j = 0; j < hStep; j++) {
          sum += img[y * hStep + j][x * wStep + i];
        }
      }
      return sum / (wStep * hStep);
    })
  );
}

const normalize = (img: number[][]) => {
  const min = Math.min(...img.map(row => Math.min(...row)));
  const max = Math.max(...img.map(row => Math.max(...row)));
  return img.map(row => row.map(v => (v - min) / (max - min)));
}

const withCooldown = (fn: () => void, cooldown: number) => {
  let lastCall = 0;
  return () => {
    const now = Date.now();
    if(now - lastCall > cooldown) {
      lastCall = now;
      fn();
    }
  }
}

const guess = () => {
  const imgData = drawCtx.getImageData(0, 0, drawCanvas.width, drawCanvas.height);
  const imgbw = Array.from({ length: drawCanvas.height }, (_, y) =>
    Array.from({ length: drawCanvas.width, }, (_, x) => {
      const i = (y * drawCanvas.width + x) * 4;
      return (imgData.data[i] + imgData.data[i + 1] + imgData.data[i + 2]) / 3 / 255;
    })
  );
  const imgbwDownsized = downsizeGrayscale(imgbw, [28, 28]);
  drawGrayscaleOnCanvas(canvas1, imgbwDownsized);
  let x: Parameter[] | Parameter[][] | Parameter[][][] = [imgbwDownsized.map(row => row.map(v => new Parameter(v)))];
  const canvasses = [canvas2, canvas3];
  for(let i = 0; i < model.layers.length; i++) {
    x = model.layers[i].forward(x);
    if(model.layers[i] instanceof Conv2d) {
      drawGrayscaleOnCanvas(canvasses.shift()!, normalize((x[0] as Parameter[][]).map(row => row.map(p => p.getValue()))));
    }
  }
  output(x as Parameter[]);
}

const guessWithCooldown = withCooldown(guess, 100);

function output(outputs: Parameter[]) {
  const g = outputs.map((p, num) => ({ p, num }));
  g.sort((a, b) => b.p.getValue() - a.p.getValue());
  guessDiv.innerText = `Guesses: \n${g.map(({ p, num }) => `${num}: ${(p.getValue() * 100).toFixed(2)}%`).join("\n")}`
}
