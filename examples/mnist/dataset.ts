

import * as fs from "fs";
import * as path from "path";
export const trainingSet = () => readImgs(
  path.join(__dirname, "..", "..", "..", "data", "MNIST", "raw", "train-images-idx3-ubyte"),
  path.join(__dirname, "..", "..", "..", "data", "MNIST", "raw", "train-labels-idx1-ubyte"),
)
export function readImgs(imgsPath: string, labelsPath: string) {
  const fbuff = fs.readFileSync(imgsPath);
  const lbuff = fs.readFileSync(labelsPath);
  const magic = fbuff.readInt32BE(0);
  const nofImages = fbuff.readInt32BE(4);
  const nofRows = fbuff.readInt32BE(8);
  const nofCols = fbuff.readInt32BE(12);
  const sz = nofRows * nofCols;
  return {
    magic,
    nofImages,
    nofRows,
    nofCols,
    images: Array.from({ length: nofImages })
      .map((_, row) => 16 + row * sz)
      .map(begin => Array.from({ length: sz }).map((_, i) => fbuff[begin + i] / 255)),
    labels: Array.from({ length: nofImages }).map((_, i) => lbuff[8 + i])
  }
}