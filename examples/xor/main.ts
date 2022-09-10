import { Sigmoid } from "../../src/act";
import { Parameter } from "../../src/autograd";
import { Linear } from "../../src/linear";
import { SGD } from "../../src/optim";
import { Sequential } from "../../src/sequential";
import { reduceSum } from "../../src/util";

const model = new Sequential(
  new Linear(2, 2),
  new Sigmoid(),
  new Linear(2, 1),
  new Sigmoid(),
);

const xs = [[0, 0], [0, 1], [1, 0], [1, 1]].map(x => x.map(x => new Parameter(x)));
const ys = [0, 1, 1, 0].map(y => new Parameter(y));

const mse = (ysPred: Parameter[], ysReal: Parameter[]) => reduceSum(ysPred.map((y, i) => y.sub(ysReal[i]).pow(2))).div(ysPred.length);
const lr = 0.1
const optim = new SGD(model.parameters(), lr);

for(let epoch = 0; epoch < 2000000; epoch++) {
  model.zero_grad();
  const yPred = xs.map(x => model.forward(x)).flat() as Parameter[];
  const loss = mse(yPred, ys);
  loss.backward();
  optim.step();
  console.log(`Epoch ${epoch}: loss ${loss.getValue()}, acc ${yPred.map(y => y.getValue()).map(y => y > 0.5 ? 1 : 0).filter((y, i) => y === ys[i].getValue()).length / ys.length}`);
}

  