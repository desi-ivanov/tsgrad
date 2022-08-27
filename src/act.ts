import { Parameter } from "./autograd";
import { Model } from "./model";

export class ReLU extends Model {
  forward = (xs: Parameter[]): Parameter[] => xs.map(x => x.relu());
}

export class Tanh extends Model {
  forward = (xs: Parameter[]): Parameter[] => xs.map(x => x.tanh());
}

export class Sigmoid extends Model {
  forward = (xs: Parameter[]): Parameter[] => xs.map(x => x.sigmoid());
}