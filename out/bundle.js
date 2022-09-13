(function () {
    'use strict';

    // inspired by https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
    class Parameter {
        constructor(value, children = [], gradfn = () => { }) {
            this.value = value;
            this.children = children;
            this.gradfn = gradfn;
            this.add = (other) => {
                const otherPar = other instanceof Parameter ? other : new Parameter(other);
                const out = new Parameter(this.value + otherPar.value, [this, otherPar], () => {
                    this.grad += out.grad;
                    otherPar.grad += out.grad;
                });
                return out;
            };
            this.sub = (other) => this.add((other instanceof Parameter ? other : new Parameter(other)).neg());
            this.neg = () => this.mul(-1);
            this.mul = (other) => {
                const otherPar = other instanceof Parameter ? other : new Parameter(other);
                const out = new Parameter(this.value * otherPar.value, [this, otherPar], () => {
                    this.grad += otherPar.value * out.grad;
                    otherPar.grad += this.value * out.grad;
                });
                return out;
            };
            this.pow = (exp) => {
                const out = new Parameter(this.value ** exp, [this], () => {
                    this.grad += exp * this.value ** (exp - 1) * out.grad;
                });
                return out;
            };
            this.div = (other) => {
                const otherPar = other instanceof Parameter ? other : new Parameter(other);
                return this.mul(otherPar.pow(-1));
            };
            this.exp = () => {
                const out = new Parameter(Math.exp(this.value), [this], () => {
                    this.grad += out.value * out.grad;
                });
                return out;
            };
            this.log = () => {
                const out = new Parameter(Math.log(this.value), [this], () => {
                    this.grad += 1 / this.value * out.grad;
                });
                return out;
            };
            this.relu = () => {
                const out = new Parameter(Math.max(0, this.value), [this], () => {
                    this.grad += out.grad * (out.value > 0 ? 1 : 0);
                });
                return out;
            };
            this.sigmoid = () => {
                const out = new Parameter(1 / (1 + Math.exp(-this.value)), [this], () => {
                    this.grad += out.grad * (out.value * (1 - out.value));
                });
                return out;
            };
            this.tanh = () => {
                const out = new Parameter(Math.tanh(this.value), [this], () => {
                    this.grad += out.grad * (1 - out.value ** 2);
                });
                return out;
            };
            this.getValue = () => this.value;
            this.setValue = (value) => { this.value = value; };
            this.getGrad = () => this.grad;
            this.backward = () => {
                const topo = [];
                const visited = new Set();
                function buildTopo(v) {
                    if (!visited.has(v)) {
                        visited.add(v);
                        v.children.forEach(c => buildTopo(c));
                        topo.push(v);
                    }
                }
                buildTopo(this);
                topo.reverse();
                this.grad = 1;
                for (const param of topo) {
                    param.gradfn();
                }
            };
            this.toString = () => `v=${this.value.toFixed(3)} g=${this.grad.toFixed(3)}`;
            this.grad = 0;
        }
        zero_grad() { this.grad = 0; }
    }

    class Model {
        constructor() {
            this.zero_grad = () => {
                this.parameters().forEach(param => param.zero_grad());
            };
            this.parameters = () => {
                return Object
                    .values(this)
                    .filter(p => p instanceof Parameter || p instanceof Model)
                    .flatMap(p => p instanceof Parameter ? [p] : p.parameters());
            };
            this.loadFromValues = (values) => {
                this.parameters().forEach((p, i) => p.setValue(values[i]));
            };
        }
        forward(xs) {
            return xs;
        }
    }

    class ReLU extends Model {
        forward(xs) {
            return xs.map(x => Array.isArray(x) ? x.map(x => x.relu()) : x.relu());
        }
    }
    class Softmax extends Model {
        forward(xs) {
            if (xs.length === 0)
                return [];
            if (Array.isArray(xs[0])) {
                return xs.map(softmax);
            }
            return softmax(xs);
        }
    }
    const softmax = (xs) => {
        const s = xs.map(x => x.exp());
        const sum = s.reduce((a, v) => a.add(v));
        return s.map(x => x.div(sum));
    };

    class Conv1d extends Model {
        constructor(kernel_size, stride, padding, initializer = () => Math.random() * 2 - 1) {
            super();
            this.kernel_size = kernel_size;
            this.stride = stride;
            this.padding = padding;
            this.initializer = initializer;
            this.parameters = () => this.w.flat();
            this.forward = (xs) => {
                const outputSize = Math.floor((xs[0].length - this.w[0].length + 2 * this.padding) / this.stride) + 1;
                const output = Array.from({ length: outputSize }, () => Array.from({ length: outputSize }, () => new Parameter(0)));
                for (let i = -this.padding, oi = 0; i < xs.length + this.padding - this.w.length + 1; i += this.stride, oi += 1) {
                    for (let j = -this.padding, oj = 0; j < xs[0].length + this.padding - this.w.length + 1; j += this.stride, oj += 1) {
                        for (let k = 0; k < this.w.length; k++) {
                            for (let l = 0; l < this.w[0].length; l++) {
                                if (0 <= i + k && i + k < xs.length && 0 <= j + l && j + l < xs[0].length) {
                                    output[oi][oj] = output[oi][oj].add(xs[i + k][j + l].mul(this.w[k][l]));
                                }
                            }
                        }
                    }
                }
                return output;
            };
            this.w = Array.from({ length: kernel_size }, () => Array.from({ length: kernel_size }, () => new Parameter(initializer())));
        }
    }

    class Flatten extends Model {
        constructor() {
            super(...arguments);
            this.forward = (xs) => xs.flat();
        }
    }

    const reduceSum = (xs) => xs.reduce((a, v) => a.add(v));

    class Perceptron extends Model {
        constructor(input_size, use_bias = true, initializer = () => Math.random() * 2 - 1) {
            super();
            this.parameters = () => this.w.concat(this.use_bias ? [this.b] : []);
            this.forward = (x) => {
                if (x.length != this.w.length)
                    throw new Error(`Input missmatch Expected ${this.w.length} channels, but got ${x.length}`);
                const r = reduceSum(x.map((v, i) => v.mul(this.w[i])));
                return this.use_bias ? [r.add(this.b)] : [r];
            };
            this.w = Array.from({ length: input_size }, () => new Parameter(initializer()));
            this.b = new Parameter(initializer());
            this.use_bias = use_bias;
        }
    }
    class Linear extends Model {
        constructor(in_channels, out_channels, use_bias = true, initializer = () => Math.random() * 2 - 1) {
            super();
            this.parameters = () => this.w.flatMap(p => p.parameters());
            this.forward = (x) => {
                if (x.length != this.in_channels)
                    throw new Error(`Input missmatch Expected ${this.in_channels} channels, but got ${x.length}`);
                const y = this.w.map(p => p.forward(x)[0]);
                if (y.length != this.w.length)
                    throw new Error(`Output missmatch. Expected ${this.w.length} outputs, but got ${y.length}`);
                return y;
            };
            this.in_channels = in_channels;
            this.w = Array.from({ length: out_channels }, () => new Perceptron(in_channels, use_bias, initializer));
        }
    }

    class Sequential extends Model {
        constructor(...layers) {
            super();
            this.parameters = () => this.layers.flatMap(l => l.parameters());
            this.forward = (xs) => this.layers.reduce((a, l) => l.forward(a), xs);
            this.layers = layers;
        }
    }

    const savedModelParams = [-0.16833165325215496, 0.5561538064013241, 0.2737854913708997, 0.0615198808622141, -0.16900302128111244, 0.6908446497003483, -0.03851886025811144, 0.04783350612500522, 0.15313503714558285, 0.2940931933848705, -0.13741094921253635, 0.21705779565262914, 0.4126217453238908, 0.08272060321232919, 0.06058289087688052, 0.2270174932630491, 0.08923597363350032, 0.038325418143396205, -0.1829436399115083, -1.1051347551015729, 0.47950041842482555, 0.2701121050145754, -0.8789197916989073, 0.10756045171934328, 0.36765246792452194, 0.5154314704639438, -2.4812218857520585, 0.8311722165747405, -0.15709614890282778, -0.6452144318721739, -0.06112157034074402, 0.9507483889105325, 0.2834966224552199, 0.034687508225355754, -0.3834511181109221, -0.10068670265286733, 0.211253629630473, 0.0379211631176595, 0.2874415937443777, 0.3958120457252484, 0.5669599052423637, 0.8609258792990697, 0.9133399456143074, 0.5808683396919051, 0.39296546767516344, 0.20049145114278982, 0.35706737008922285, 0.42193522414148443, -0.14703142829955915, -0.9238195769199181, 0.18083577287140853, -3.0727426565084137, 0.4272318932271357, 0.2051759721395223, -1.2287150391665163, 0.29823382394747955, -1.559047214021963, 1.025634694221277, 0.4595988242105882, 0.6927121237709462, -28.121070037283143, -4.003787586773895, -0.2848795533112128, -4.059574715961078, -2.888613837271923, -5.264105412931792, 1.3035511825397468, -2.0745112915957304, 2.8065628276673427, -7.1341067236667435, -1.2330591972433531, 0.7003471643489858, 0.9741721607040297, -4.902996334062544, 0.8111588853224959, -0.24972740744191882, 0.03614782160435822, -12.934913815614593, 0.32474841192209997, -2.3860095150599108, 1.3273265447782678, -0.24142593844038351, 0.5902783259295811, 0.3603146153006253, -1.765403199752041, -2.2859064209506537, 1.0092997992852721, -3.452827494053766, 1.0429761116894363, 0.33536579099574065, 2.5737706338340574, -0.41076603520192817, 0.18173531769435133, -1.3068587895602484, -9.96390480709427, -5.008581730553151, -4.417017339433912, -3.033958535994055, 2.098316216131912, 0.5790515674373188, -0.9232522133207005, -1.0411753391926137, -1.5450758857202456, -3.635475740149299, -3.056914512857573, 0.07423014911376362, 0.7144741960294365, 2.2188519528855193, -2.5523486397165493, -0.16131543863322143, 0.8524578360978483, 0.016005908073218122, -1.8550717742000102, 1.134959986930942, -2.381162799420524, 0.8905432770246711, -0.9462080771246685, -1.1755880352470804, -0.982602742028892, 2.147792295973834, 0.6462961470740921, 1.100214894753029, 0.4341161538594833, -0.7640132772129223, -4.7998469636690935, -2.54931244004527, -0.3848886211106119, 0.04505772634148938, -2.7406210206689914, 1.306152306685068, -0.7667223518349897, 0.9604183638636775, -1.5326099567600082, 1.9887753929376903, -1.919925871860269, 1.2697290644089425, 1.0438709027147814, -1.3439321448121908, 0.22011662107814767, -3.057047317448933, -1.726480172747212, -0.4171466459114652, -1.5920020919490403, 0.3463946493686121, 0.18752334664689174, 1.56094181941111, -0.9593986850542211, 0.3515048244789258, -0.22899664436867667, 3.254325721068535, -2.4155364575940887, 1.5908389119554665, 0.5203343332477085, 4.669432578167987, -1.4133778830949704, 1.3092229755322422, -0.12741872207803123, -3.5245316521840167, -2.7619633298172466, -0.9599397480916626, -0.39857755361797137, 0.8673611585501961, -26.159177686990454, -5.986146299286394, -2.3647187708987496, -3.636479136597178, -2.276739029058617, -0.6282572336759567, -4.199401923572759, -3.582688053928491, -2.8204273443162906, -0.5000868817890266, -2.3765345092677403, 0.031685015368742915, 0.36228203612734516, -4.664383073858481, 1.5256451317564461, 0.35246316576914233, 0.2825671685236274, 1.8408445943171778, 1.0115017930475252, 0.7906505392143474, 1.4756023588718372, 0.8718153421768986, 1.0748230637397647, -0.5708706931264065, 0.6288239790982021, -1.2308779431832726, -0.276808308905523, -2.427501914239578, -0.8286781845780099, 1.2499427591483057, -0.49725234795382633, -0.6793619832401079, 0.1776382653759765, -1.116929617323106, -5.258618856788029, -1.1167923928525452, -0.33472945127431253, 1.399742398243244, -0.24349717729735984, -1.9677856778796345, -3.5997114807859054, -0.20479934915598447, -6.544352725643844, 3.4335089041295963, -1.9717739389988482, 0.9233951153278962, -1.4114883071640563, -4.761675890240232, -2.301400187145512, 0.653353480089295, -0.5913695100178314, 3.1105824291447752, 2.9886991974644705, -0.162925311607531, 2.29484873684672, -0.19769783195642615];

    const drawCanvas = document.getElementById("draw");
    const canvas1 = document.getElementById("display1");
    const canvas2 = document.getElementById("display2");
    const canvas3 = document.getElementById("display3");
    const canvas4 = document.getElementById("display4");
    const resetButton = document.getElementById("btn-reset");
    const guessDiv = document.getElementById("result-guess");
    const pixelMultiplier = 14;
    [drawCanvas, canvas1, canvas2, canvas3, canvas4].forEach(canvas => {
        canvas.width = 28 * pixelMultiplier;
        canvas.height = 28 * pixelMultiplier;
    });
    const drawCtx = drawCanvas.getContext("2d");
    resetButton.addEventListener("click", () => {
        [drawCanvas, canvas1, canvas2, canvas3, canvas4].forEach(canvas => {
            const ctx = canvas.getContext("2d");
            ctx.fillStyle = "#000";
            ctx.fillRect(0, 0, canvas.width, canvas.height);
        });
    });
    drawCtx.fillStyle = "#000";
    drawCtx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
    let mousedown = false;
    drawCanvas.addEventListener("mousedown", e => {
        mousedown = true;
        drawCtx.fillStyle = "#fff";
        drawCtx.strokeStyle = "#fff";
        drawCtx.beginPath();
        drawCtx.moveTo(e.offsetX, e.offsetY);
    });
    document.body.addEventListener("mouseup", e => {
        mousedown = false;
        drawCtx.closePath();
    });
    drawCanvas.addEventListener("mousemove", e => {
        if (mousedown) {
            drawCtx.lineWidth = pixelMultiplier * 3;
            drawCtx.lineCap = "round";
            drawCtx.lineJoin = "round";
            drawCtx.lineTo(e.offsetX, e.offsetY);
            drawCtx.stroke();
            drawCtx.moveTo(e.offsetX, e.offsetY);
            guess();
        }
    });
    const model = new Sequential(new Conv1d(4, 2, 1), new ReLU(), new Conv1d(4, 2, 1), new ReLU(), new Conv1d(4, 2, 2), new ReLU(), new Flatten(), new Linear(16, 10), new Softmax());
    model.loadFromValues(savedModelParams);
    const drawGrayscaleOnCanvas = (canvas, img) => {
        const ctx = canvas.getContext("2d");
        const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const repeat = Math.ceil(canvas.width / img[0].length);
        for (let y = 0; y < img.length; y++) {
            for (let x = 0; x < img[0].length; x++) {
                const val = img[y][x];
                for (let i = 0; i < repeat; i++) {
                    for (let j = 0; j < repeat; j++) {
                        const idx = (y * repeat + j) * canvas.width * 4 + (x * repeat + i) * 4;
                        imgData.data[idx] = val * 255;
                        imgData.data[idx + 1] = val * 255;
                        imgData.data[idx + 2] = val * 255;
                        imgData.data[idx + 3] = 255;
                    }
                }
            }
        }
        ctx.putImageData(imgData, 0, 0);
    };
    const downsizeGrayscale = (img, shape) => {
        const [width, height] = shape;
        const wStep = Math.ceil(img[0].length / width);
        const hStep = Math.ceil(img.length / height);
        return Array.from({ length: height }, (_, y) => Array.from({ length: width }, (_, x) => {
            let sum = 0;
            for (let i = 0; i < wStep; i++) {
                for (let j = 0; j < hStep; j++) {
                    sum += img[y * hStep + j][x * wStep + i];
                }
            }
            return sum / (wStep * hStep);
        }));
    };
    const normalize = (img) => {
        const min = Math.min(...img.map(row => Math.min(...row)));
        const max = Math.max(...img.map(row => Math.max(...row)));
        return img.map(row => row.map(v => (v - min) / (max - min)));
    };
    function guess() {
        const imgData = drawCtx.getImageData(0, 0, drawCanvas.width, drawCanvas.height);
        const imgbw = Array.from({ length: drawCanvas.height }, (_, y) => Array.from({ length: drawCanvas.width, }, (_, x) => {
            const i = (y * drawCanvas.width + x) * 4;
            return (imgData.data[i] + imgData.data[i + 1] + imgData.data[i + 2]) / 3 / 255;
        }));
        const imgbwDownsized = downsizeGrayscale(imgbw, [28, 28]);
        drawGrayscaleOnCanvas(canvas1, imgbwDownsized);
        let x = imgbwDownsized.map(row => row.map(v => new Parameter(v)));
        const canvasses = [canvas2, canvas3, canvas4];
        for (let i = 0; i < model.layers.length; i++) {
            x = model.layers[i].forward(x);
            if (model.layers[i] instanceof Conv1d) {
                drawGrayscaleOnCanvas(canvasses.shift(), normalize(x.map(row => row.map(p => p.getValue()))));
            }
        }
        output(x);
    }
    function output(outputs) {
        const g = outputs.map((p, num) => ({ p, num }));
        g.sort((a, b) => b.p.getValue() - a.p.getValue());
        guessDiv.innerText = `Guesses: \n${g.map(({ p, num }) => `${num}: ${(p.getValue() * 100).toFixed(2)}%`).join("\n")}`;
    }

})();
