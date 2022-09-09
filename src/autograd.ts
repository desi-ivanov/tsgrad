export class Parameter {
  private grad: number
  constructor(
    private value: number,
    public readonly children: Parameter[] = [],
    private readonly backprop: (grad: number) => void = () => { }
  ) {
    this.grad = 0
  }

  add = (other: Parameter | number) => {
    const otherPar = other instanceof Parameter ? other : new Parameter(other)
    return new Parameter(
      this.value + otherPar.value,
      [this, otherPar],
      (grad: number) => {
        this.grad += grad
        otherPar.grad += grad
      }
    )
  }

  sub = (other: Parameter | number) => this.add((other instanceof Parameter ? other : new Parameter(other)).neg())
  neg = () => this.mul(-1)
  mul = (other: Parameter | number) => {
    const otherPar = other instanceof Parameter ? other : new Parameter(other)
    return new Parameter(
      this.value * otherPar.value,
      [this, otherPar],
      (grad: number) => {
        this.grad += grad * otherPar.value
        otherPar.grad += grad * this.value
      }
    )
  }
  pow = (exp: number) => {
    return new Parameter(
      this.value ** exp,
      [this],
      (grad: number) => {
        this.grad += grad * this.value ** (exp - 1) * exp
      }
    )
  }
  div = (other: Parameter | number) => {
    const otherPar = other instanceof Parameter ? other : new Parameter(other)
    return new Parameter(
      this.value / otherPar.value,
      [this, otherPar],
      (grad: number) => {
        this.grad += grad / otherPar.value
        otherPar.grad -= grad * this.value / otherPar.value ** 2
      }
    )
  }
  exp = () => {
    const v = Math.exp(this.value);
    return new Parameter(
      v,
      [this],
      (grad: number) => {
        this.grad += grad * v
      }
    );
  }
  log = () => {
    return new Parameter(
      Math.log(this.value),
      [this],
      (grad: number) => {
        this.grad += grad / this.value
      }
    );
  }
  relu = () => {
    const out = new Parameter(
      Math.max(0, this.value),
      [this],
      (grad: number) => {
        this.grad += grad * (out.value > 0 ? 1 : 0)
      }
    );
    return out;
  }
  sigmoid = () => {
    const out = new Parameter(
      1 / (1 + Math.exp(-this.value)),
      [this],
      (grad: number) => {
        this.grad += grad * (out.value * (1 - out.value))
      }
    )
    return out;
  }
  tanh = () => {
    const out = new Parameter(
      Math.tanh(this.value),
      [this],
      (grad: number) => {
        this.grad += grad * (1 - out.value ** 2)
      }
    )
    return out;
  }
  getValue = () => this.value
  setValue = (value: number) => { this.value = value }
  zero_grad() { this.grad = 0 }
  backward = () => {
    const topo: Parameter[] = [];
    const visited = new Set<Parameter>();
    const stack: Parameter[] = [this];
    while(stack.length > 0) {
      const current = stack.pop()!;
      if(visited.has(current)) {
        continue;
      }
      visited.add(current);
      topo.push(current);
      for(const dep of current.children) {
        stack.push(dep);
      }
    }
    this.grad = 1;
    for(const param of topo) {
      param.backprop(param.grad);
    }
  }
  update = (lr: number) => {
    this.value -= lr * this.grad;
  }
  toString = () => `v=${this.value} g=${this.grad}`
}