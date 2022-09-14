import { Parameter } from "./autograd"
import { Model } from "./model";

export class Dropout extends Model {
  constructor(
    public readonly p: number
  ) { super() }
  forward(xs: Parameter[]): Parameter[];
  forward(xs: Parameter[][]): Parameter[][];
  forward(xs: Parameter[][] | Parameter[]) {
    return xs.map(x => Array.isArray(x) ? this.forward(x) : x.mul(new Parameter(Math.random() > this.p ? 1 : 0)));
  }
}