import { ReLU, Sigmoid } from "../../src/act";
import { Parameter } from "../../src/autograd";
import { Linear } from "../../src/linear";
import { Sequential } from "../../src/sequential";

const model = new Sequential(
  new Linear(2, 10),
  new ReLU(),
  new Linear(10, 1),
  new Sigmoid()
);

const xs = [[0, 0], [0, 1], [1, 0], [1, 1]].map(x => x.map(x => new Parameter(x)));
const ys = [0, 1, 1, 0].map(y => new Parameter(y));

const computeLoss = (ysPred: Parameter[], ysReal: Parameter[]) => ysPred.map((y, i) => y.sub(ysReal[i]).pow(2)).reduce((a, v) => a.add(v));

const learning_rate = 0.01

for(let epoch = 0; epoch < 2000000; epoch++) {
  model.zero_grad();
  const yPred = xs.map(x => model.forward(x)).flat() as Parameter[];
  const loss = computeLoss(yPred, ys);
  loss.backward();
  model.parameters().forEach(p => p.update(learning_rate));
  console.log(`Epoch ${epoch}: loss ${loss.getValue()}, acc ${yPred.map(y => y.getValue()).map(y => y > 0.5 ? 1 : 0).filter((y, i) => y === ys[i].getValue()).length / ys.length}`);
}