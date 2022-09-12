// inspired by https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
export class Parameter {
  private grad: number
  constructor(
    private value: number,
    public readonly children: Parameter[] = [],
    private readonly gradfn: () => void = () => { },
  ) {
    this.grad = 0
  }

  add = (other: Parameter | number) => {
    const otherPar = other instanceof Parameter ? other : new Parameter(other)
    const out = new Parameter(
      this.value + otherPar.value,
      [this, otherPar],
      () => {
        this.grad += out.grad
        otherPar.grad += out.grad
      },
    );
    return out;
  }

  sub = (other: Parameter | number) => this.add((other instanceof Parameter ? other : new Parameter(other)).neg())
  neg = () => this.mul(-1)
  mul = (other: Parameter | number) => {
    const otherPar = other instanceof Parameter ? other : new Parameter(other)
    const out = new Parameter(
      this.value * otherPar.value,
      [this, otherPar],
      () => {
        this.grad += otherPar.value * out.grad
        otherPar.grad += this.value * out.grad
      },
    );
    return out;
  }
  pow = (exp: number) => {
    const out = new Parameter(
      this.value ** exp,
      [this],
      () => {
        this.grad += exp * this.value ** (exp - 1) * out.grad;
      }
    );
    return out;
  }
  div = (other: Parameter | number) => {
    const otherPar = other instanceof Parameter ? other : new Parameter(other)
    return this.mul(otherPar.pow(-1));
  }
  exp = () => {
    const out = new Parameter(
      Math.exp(this.value),
      [this],
      () => {
        this.grad += out.value * out.grad;
      }
    );
    return out;
  }
  log = () => {
    const out = new Parameter(
      Math.log(this.value),
      [this],
      () => {
        this.grad += 1 / this.value * out.grad;
      }
    );
    return out;
  }
  relu = () => {
    const out = new Parameter(
      Math.max(0, this.value),
      [this],
      () => {
        this.grad += out.grad * (out.value > 0 ? 1 : 0)
      }
    );
    return out;
  }
  sigmoid = () => {
    const out = new Parameter(
      1 / (1 + Math.exp(-this.value)),
      [this],
      () => {
        this.grad += out.grad * (out.value * (1 - out.value))
      }
    )
    return out;
  }
  tanh = () => {
    const out = new Parameter(
      Math.tanh(this.value),
      [this],
      () => {
        this.grad += out.grad * (1 - out.value ** 2)
      }
    )
    return out;
  }
  getValue = () => this.value
  setValue = (value: number) => { this.value = value }
  getGrad = () => this.grad
  zero_grad() { this.grad = 0 }
  backward = () => {
    const topo: Parameter[] = [];
    const visited = new Set<Parameter>();
    function buildTopo(v: Parameter) {
      if(!visited.has(v)) {
        visited.add(v)
        v.children.forEach(c => buildTopo(c));
        topo.push(v)
      }
    }
    buildTopo(this)
    topo.reverse();

    this.grad = 1;
    for(const param of topo) {
      param.gradfn();
    }
  }
  toString = () => `v=${this.value.toFixed(3)} g=${this.grad.toFixed(3)}`
}