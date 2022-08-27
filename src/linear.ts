import { Parameter } from "./autograd";
import { Model } from "./model";

export class Linear extends Model {
  w: Parameter[][];
  b: Parameter | undefined;
  in_channels: number;
  constructor(in_channels: number, out_channels: number, use_bias = true, initializer = () => Math.random() * 2 - 1) {
    super();
    this.in_channels = in_channels
    this.w = Array.from({ length: out_channels }, () => Array.from({ length: in_channels }, () => new Parameter(initializer())));
    if(use_bias) {
      this.b = new Parameter(initializer());
    }
  }
  parameters = () => this.w.flat().concat(this.b ? [this.b] : []);
  forward = (x: Parameter[]): Parameter[] => {
    if(x.length != this.in_channels) throw new Error(`Input missmatch Expected ${this.in_channels} channels, but got ${x.length}`);
  
    const y = this.w.map(ws => ws.map((w, i) => w.mul(x[i])).reduce((a, v) => a.add(v)).add(this.b ? this.b : 0));

    if(y.length != this.w.length) throw new Error(`Output missmatch. Expected ${this.w.length} outputs, but got ${y.length}`);
    return y;
  }

}