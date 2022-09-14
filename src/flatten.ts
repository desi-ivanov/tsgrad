import { Parameter } from "./autograd";
import { Model } from "./model";

export class Flatten extends Model {
  forward = (xs: Parameter[][] | Parameter[]): Parameter[] => xs.flat(5);
}