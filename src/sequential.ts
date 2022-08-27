import { Parameter } from "./autograd";
import { Model } from "./model";

export class Sequential extends Model {
  layers: Model[];
  constructor(...layers: Model[]) {
    super();
    this.layers = layers;
  }
  parameters = () => this.layers.flatMap(l => l.parameters());
  forward = (xs: Parameter[]) => this.layers.reduce((a, l) => l.forward(a), xs);
}
