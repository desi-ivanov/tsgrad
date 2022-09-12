import { Parameter } from "./autograd";
import { Model } from "./model";
import { reduceSum } from "./util";

class Perceptron extends Model {
  w: Parameter[];
  b: Parameter;
  use_bias: boolean;
  constructor(input_size: number, use_bias = true, initializer = () => Math.random() * 2 - 1) {
    super();
    this.w = Array.from({ length: input_size }, () => new Parameter(initializer()));
    this.b = new Parameter(initializer());
    this.use_bias = use_bias 
  }
  parameters = () => this.w.concat(this.use_bias ? [this.b] : []);
  forward = (x: Parameter[]): Parameter[] => {
    if(x.length != this.w.length) throw new Error(`Input missmatch Expected ${this.w.length} channels, but got ${x.length}`);
    const r = reduceSum(x.map((v, i) => v.mul(this.w[i])))
    return this.use_bias ? [r.add(this.b)] : [r];
  }
}

export class Linear extends Model {
  w: Perceptron[];
  b: Parameter | undefined;
  in_channels: number;
  constructor(in_channels: number, out_channels: number, use_bias = true, initializer = () => Math.random() * 2 - 1) {
    super();
    this.in_channels = in_channels
    this.w = Array.from({ length: out_channels }, () => new Perceptron(in_channels, use_bias, initializer));
  }
  parameters = () => this.w.flatMap(p => p.parameters());
  forward = (x: Parameter[]): Parameter[] => {
    if(x.length != this.in_channels) throw new Error(`Input missmatch Expected ${this.in_channels} channels, but got ${x.length}`);
    const y = this.w.map(p => p.forward(x)[0]);
    if(y.length != this.w.length) throw new Error(`Output missmatch. Expected ${this.w.length} outputs, but got ${y.length}`);
    return y;
  }

}