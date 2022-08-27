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