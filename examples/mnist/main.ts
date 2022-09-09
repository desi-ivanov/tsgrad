import { ReLU, Sigmoid, Softmax } from "../../src/act";
import { Parameter } from "../../src/autograd";
import { Conv1d } from "../../src/conv";
import { Flatten } from "../../src/flatten";
import { Linear } from "../../src/linear";
import { Sequential } from "../../src/sequential";
import { argmax, zip } from "../../src/util";
import { trainingSet } from "./dataset";

const model = new Sequential(
  new Conv1d(4, 2, 1),
  new ReLU(),
  new Conv1d(4, 2, 1),
  new ReLU(),
  new Conv1d(4, 2, 2),
  new ReLU(),
  new Flatten(),
  new Linear(16, 10, true),
  new Softmax(),
);

const data = trainingSet();
const sq = Math.floor(data.images[0].length ** 0.5)

const efficientCrossEntropy = (ysPred: Parameter[], y: number) => ysPred[y].log().neg();

const xss = data.images.slice(0, 10000);
const yss = data.labels.slice(0, 10000);

const batchSize = 16;
const learningRate = 0.01;

for(let epoch = 0; epoch < 2000000; epoch++) {
  const preds: number[] = [];
  const reals: number[] = [];
  const losses: number[] = [];
  for(let i = 0; i < (xss.length / batchSize - 1); i++) {
    model.zero_grad();
    for(let j = 0; j < batchSize; j++) {
      const x = xss[i * batchSize + j];
      const y = yss[i * batchSize + j];
      const xParam = Array.from({ length: sq }, (_, i) => x.slice(i * sq, i * sq + sq)).map(xs => xs.map(x => new Parameter(x)));

      const yPred = model.forward(xParam) as Parameter[];
      const loss = efficientCrossEntropy(yPred, y);
      loss.backward();

      preds.push(argmax(yPred.map(p => p.getValue())));
      reals.push(y);
      losses.push(loss.getValue());
    }
    model.parameters().forEach(p => p.update(learningRate / batchSize));
  }
  console.log(`Epoch ${epoch} acc: ${zip(preds, reals).filter(([p, r]) => p === r).length / preds.length}, loss ${losses.reduce((a, v) => a + v, 0) / losses.length}`);
}
