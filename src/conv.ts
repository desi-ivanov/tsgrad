import { Parameter } from "./autograd";
import { Model } from "./model";

export class Conv2d extends Model {
  private readonly w: Parameter[][];
  constructor(
    public readonly kernel_size: number,
    public readonly stride: number,
    public readonly padding: number,
    public readonly initializer: (() => number) = () => Math.random() * 2 - 1
  ) {
    super();
    this.w = Array.from({ length: kernel_size }, () => Array.from({ length: kernel_size }, () => new Parameter(initializer())));
  }
  parameters = () => this.w.flat();
  forward = (xs: Parameter[][]): Parameter[][] => {
    const outputSize = Math.floor((xs[0].length - this.w[0].length + 2 * this.padding) / this.stride) + 1;
    const output = Array.from({ length: outputSize }, () => Array.from({ length: outputSize }, () => new Parameter(0)));
    for(let i = -this.padding, oi = 0; i < xs.length + this.padding - this.w.length + 1; i += this.stride, oi += 1) {
      for(let j = -this.padding, oj = 0; j < xs[0].length + this.padding - this.w.length + 1; j += this.stride, oj += 1) {
        for(let k = 0; k < this.w.length; k++) {
          for(let l = 0; l < this.w[0].length; l++) {
            if(0 <= i + k && i + k < xs.length && 0 <= j + l && j + l < xs[0].length) {
              output[oi][oj] = output[oi][oj].add(xs[i + k][j + l].mul(this.w[k][l]));
            }
          }
        }
      }
    }

    return output;
  }
}
