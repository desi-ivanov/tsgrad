# tsgrad
A simple automatic differentiation library
## Example
```ts
import { ReLU, Softmax } from "./src/act";
import { Adam } from "./src/optim"
import { Parameter } from "./src/autograd";
import { Conv1d } from "./src/conv";
import { Flatten } from "./src/flatten";
import { Linear } from "./src/linear";
import { Sequential } from "./src/sequential";
import { argmax, reduceSum, zip } from "./src/util";
import { trainingSet } from "./examples/mnist/dataset";
import { Model } from "./src/model";

const model = new Sequential(
  new Conv1d(4, 2, 1),
  new ReLU(),
  new Conv1d(4, 2, 1),
  new ReLU(),
  new Conv1d(4, 2, 2),
  new ReLU(),
  new Flatten(),
  new Linear(16, 10),
  new Softmax(),
);

const data = trainingSet();
const sq = Math.floor(data.images[0].length ** 0.5)

const crossEntropy = (ysPred: Parameter[], y: number) => 
  reduceSum(ysPred.map((yPred, i) => yPred.log().mul(y === i ? 1 : 0))).neg();

const xss = data.images;
const yss = data.labels;

const batchSize = 32;
const learningRate = 0.01;

const optim = new Adam(model.parameters(), learningRate);

for(let epoch = 0; epoch < 50; epoch++) {
  for(let i = 0; i < (xss.length / batchSize - 1); i++) {

    model.zero_grad();
    for(let j = 0; j < batchSize; j++) {
      const x = xss[i * batchSize + j];
      const y = yss[i * batchSize + j];
      const xParam = Array
        .from({ length: sq }, (_, i) => x.slice(i * sq, i * sq + sq))
        .map(xs => xs.map(x => new Parameter(x)));
      const yPred = model.forward(xParam) as Parameter[];
      const loss = crossEntropy(yPred, y);
      loss.backward();
    }

    optim.step();
  }
}

```

## Credits
Highly inspired by torch's [autograd](https://pytorch.org/docs/stable/autograd.html) and Karpathy's [micrograd](https://github.com/karpathy/micrograd).

# License

MIT
