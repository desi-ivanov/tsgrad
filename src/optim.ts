import { Parameter } from "./autograd"

export class SGD {
  constructor(
    private readonly parameters: Parameter[],
    private readonly learningRate: number
  ) { }

  step = () => {
    this.parameters.forEach(p => {
      p.setValue(p.getValue() - this.learningRate * p.getGrad());
    })
  }
}

export class Adam {
  constructor(
    private readonly parameters: Parameter[],
    private readonly learningRate: number,
    private readonly beta1: number = 0.9,
    private readonly beta2: number = 0.999,
    private readonly epsilon: number = 1e-8
  ) { }

  private readonly m: number[] = this.parameters.map(p => 0)
  private readonly v: number[] = this.parameters.map(p => 0)
  private t = 0

  step = () => {
    this.t += 1
    this.parameters.forEach((p, i) => {
      this.m[i] = this.beta1 * this.m[i] + (1 - this.beta1) * p.getGrad()
      this.v[i] = this.beta2 * this.v[i] + (1 - this.beta2) * p.getGrad() ** 2
      const mHat = this.m[i] / (1 - this.beta1 ** this.t)
      const vHat = this.v[i] / (1 - this.beta2 ** this.t)
      p.setValue(p.getValue() - this.learningRate * mHat / (vHat ** 0.5 + this.epsilon))
    })
  }

}