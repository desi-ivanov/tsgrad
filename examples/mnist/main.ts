import { ReLU, Softmax } from "../../src/act";
import { Adam } from "../../src/optim"
import { Parameter } from "../../src/autograd";
import { Conv2d } from "../../src/conv";
import { Flatten } from "../../src/flatten";
import { Linear } from "../../src/linear";
import { Sequential } from "../../src/sequential";
import { argmax, reduceSum, zip } from "../../src/util";
import { trainingSet } from "./dataset";
import * as ProgressBar from "progress";
import { Model } from "../../src/model";
import * as fs from "fs";
import { Dropout } from "../../src/dropout";
const saveModel = (model: Model, path: string) => fs.writeFileSync(path, JSON.stringify(model.parameters().map(p => (p.getValue()))));
const loadModel = (model: Model, path: string) => model.loadFromValues(JSON.parse(fs.readFileSync(path, "utf8")));

const model = new Sequential(
  new Conv2d(4, 3, 2, 0),
  new Dropout(0.3),
  new ReLU(),
  new Conv2d(4, 3, 2, 0),
  new Dropout(0.3),
  new ReLU(),
  new Flatten(),
  new Linear(144, 10),
  new Softmax(),
);

const data = trainingSet();
const sq = Math.floor(data.images[0].length ** 0.5)

const crossEntropy = (ysPred: Parameter[], y: number) => reduceSum(ysPred.map((yPred, i) => yPred.log().mul(y === i ? 1 : 0))).neg();

const xss = data.images;
const yss = data.labels;

const batchSize = 32;
const learningRate = 0.01;

const optim = new Adam(model.parameters(), learningRate);
loadModel(model, "model3.json");
for(let epoch = 0; epoch < 2000000; epoch++) {
  const preds: number[] = [];
  const reals: number[] = [];
  const losses: number[] = [];
  const pb = new ProgressBar(":bar", { total: Math.floor(xss.length / batchSize) });
  for(let i = 0; i < (xss.length / batchSize - 1); i++) {
    model.zero_grad();
    for(let j = 0; j < batchSize; j++) {
      const x = xss[i * batchSize + j];
      const y = yss[i * batchSize + j];
      const xParam = Array.from({ length: sq }, (_, i) => x.slice(i * sq, i * sq + sq)).map(xs => xs.map(x => new Parameter(x)));
      const yPred = model.forward([xParam]) as Parameter[];
      const loss = crossEntropy(yPred, y);
      loss.div(batchSize).backward();
      preds.push(argmax(yPred.map(p => p.getValue())));
      reals.push(y);
      losses.push(loss.getValue());
    }
    pb.tick();
    optim.step();
  }
  console.log(`Epoch ${epoch} acc: ${zip(preds, reals).filter(([p, r]) => p === r).length / preds.length}, loss ${losses.reduce((a, v) => a + v, 0) / losses.length}`);
  saveModel(model, "model3.json");
}
