import { Parameter } from "./autograd";

export class Model {
  zero_grad = () => {
    this.parameters().forEach(param => param.zero_grad())
  }
  parameters = (): Parameter[] => {
    return Object
      .values<Parameter | Model>(this as any)
      .filter(p => p instanceof Parameter || p instanceof Model)
      .flatMap(p => p instanceof Parameter ? [p] : p.parameters());
  }
  forward(xs: Parameter[] | Parameter[][] | Parameter[][][]): Parameter[] | Parameter[][] | Parameter[][][] {
    return xs;
  }
  loadFromValues = (values: number[]) => {
    this.parameters().forEach((p, i) => p.setValue(values[i])); 
  }
}
