import { Parameter } from "./autograd";
import { Model } from "./model";

export class ReLU extends Model {
  forward(xs: Parameter[]): Parameter[];
  forward(xs: Parameter[][]): Parameter[][];
  forward(xs: Parameter[] | Parameter[][]) {
    return xs.map(x => Array.isArray(x) ? x.map(x => x.relu()) : x.relu());
  }

}

export class Tanh extends Model {
  forward(xs: Parameter[]): Parameter[];
  forward(xs: Parameter[][]): Parameter[][];
  forward(xs: Parameter[] | Parameter[][]) {
    return xs.map(x => Array.isArray(x) ? x.map(x => x.tanh()) : x.tanh());
  }
}

export class Sigmoid extends Model {
  forward(xs: Parameter[]): Parameter[];
  forward(xs: Parameter[][]): Parameter[][];
  forward(xs: Parameter[] | Parameter[][]) {
    return xs.map(x => Array.isArray(x) ? x.map(x => x.sigmoid()) : x.sigmoid());
  }
}
export class Softmax extends Model {
  forward(xs: Parameter[]): Parameter[];
  forward(xs: Parameter[][]): Parameter[][];
  forward(xs: Parameter[] | Parameter[][]) {
    if(xs.length === 0) return []
    if(Array.isArray(xs[0])) {
      return (xs as Parameter[][]).map(softmax);
    }
    return softmax(xs as Parameter[]);
  }
}

const softmax = (xs: Parameter[]) => {
  const s = xs.map(x => x.exp());
  const sum = s.reduce((a, v) => a.add(v));
  return s.map(x => x.div(sum));
}
