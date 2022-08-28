import { Parameter } from "./autograd";

export const zip = <T, U>(xs: T[], ys: U[]): [T, U][] => xs.map((x, i) => [x, ys[i]]);
export const argmax = (xs: number[]) => xs.reduce((a, v, i) => v > xs[a] ? i : a, 0);
export const reduceSum = (xs: Parameter[]) => xs.reduce((a, v) => a.add(v));