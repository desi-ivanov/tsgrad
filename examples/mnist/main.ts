import { ReLU, Softmax } from "../../src/act";
import { Parameter } from "../../src/autograd";
import { Conv2d } from "../../src/conv";
import { Flatten } from "../../src/flatten";
import { Linear } from "../../src/linear";
import { Sequential } from "../../src/sequential";
import { argmax, reduceSum, zip } from "../../src/util";
import { trainingSet } from "./dataset";

const data = trainingSet();

const model = new Sequential(
  new Conv2d(4, 2, 1),
  new ReLU(),
  new Conv2d(4, 2, 1),
  new ReLU(),
  new Conv2d(4, 2, 2),
  new ReLU(),
  new Flatten(),
  new Linear(16, 10),
  new Softmax(),
);

const sq = Math.floor(data.images[0].length ** 0.5)

const crossEntropy = (ysPred: Parameter[], ysReal: Parameter[]) => reduceSum(zip(ysPred, ysReal).map(([p, t]) => t.mul(p.log())));

const xss = data.images.slice(0, 1000).map(xs => Array.from({ length: sq }, (_, i) => xs.slice(i * sq, i * sq + sq)).map(xs => xs.map(x => new Parameter(x))));
const yss = data.labels.slice(0, 1000).map(y => Array.from({ length: 10 }, (_, i) => new Parameter(y === i ? 1 : 0)));
const batch_size = 64;
const learning_rate = 0.001;
const criterion = crossEntropy;
for(let epoch = 0; epoch < 2000000; epoch++) {
  const preds: number[] = [];
  const reals: number[] = [];
  const losses: number[] = [];
  for(let i = 0; i < (xss.length / batch_size - 1); i++) {
    model.zero_grad();
    for(let j = 0; j < batch_size; j++) {
      const x = xss[i * batch_size + j];
      const y = yss[i * batch_size + j];
      const yPred = model.forward(x) as Parameter[];
      preds.push(argmax(yPred.map(p => p.getValue())));
      reals.push(argmax(y.map(p => p.getValue())));
      const loss = criterion(yPred, y)
      loss.backward();
      losses.push(loss.getValue());
    }
    model.parameters().forEach(p => p.update(learning_rate));
  }
  console.log(`Epoch ${epoch} acc: ${zip(preds, reals).filter(([p, r]) => p === r).length / preds.length}, loss ${losses.reduce((a, v) => a + v, 0) / losses.length}`);
}
