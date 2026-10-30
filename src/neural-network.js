(function (root, factory) {
    if (typeof module === 'object' && module.exports) {
        module.exports = factory();
    } else {
        root.NeuralNetwork = factory();
    }
}(typeof self !== 'undefined' ? self : this, function () {
    'use strict';

    var LEAKY_ALPHA = 0.01;

    function fail(message) {
        throw new Error('NeuralNetwork: ' + message);
    }

    function randn() {
        var u = 0;
        var v = 0;
        while (u === 0) {
            u = Math.random();
        }
        while (v === 0) {
            v = Math.random();
        }
        return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    }

    function leakyRelu(z) {
        return z > 0 ? z : LEAKY_ALPHA * z;
    }

    function leakyReluDerivative(z) {
        return z > 0 ? 1 : LEAKY_ALPHA;
    }

    function softmax(zs) {
        var max = zs[0];
        var i;
        for (i = 1; i < zs.length; i++) {
            if (zs[i] > max) {
                max = zs[i];
            }
        }
        var exps = [];
        var sum = 0;
        for (i = 0; i < zs.length; i++) {
            var e = Math.exp(zs[i] - max);
            exps.push(e);
            sum += e;
        }
        var out = [];
        for (i = 0; i < exps.length; i++) {
            out.push(exps[i] / sum);
        }
        return out;
    }

    function copyVector(src) {
        var out = [];
        for (var i = 0; i < src.length; i++) {
            out.push(src[i]);
        }
        return out;
    }

    function Network(options) {
        if (!options || !options.layers || options.layers.length < 2) {
            fail('create() requires options.layers with at least input and output sizes');
        }
        var layers = options.layers;
        var i;
        for (i = 0; i < layers.length; i++) {
            if (typeof layers[i] !== 'number' || layers[i] < 1 || layers[i] !== Math.floor(layers[i])) {
                fail('each layer size must be a positive integer');
            }
        }
        this.layers = layers.slice();
        this.activation = options.activation || 'leakyRelu';
        this.outputActivation = options.outputActivation || 'linear';
        if (this.activation !== 'leakyRelu') {
            fail('unsupported activation "' + this.activation + '"');
        }
        if (this.outputActivation !== 'linear' && this.outputActivation !== 'softmax') {
            fail('unsupported outputActivation "' + this.outputActivation + '"');
        }
        this.weights = [];
        this.biases = [];
        this._initWeights();
    }

    Network.prototype._initWeights = function () {
        this.weights = [];
        this.biases = [];
        for (var l = 1; l < this.layers.length; l++) {
            var fanIn = this.layers[l - 1];
            var fanOut = this.layers[l];
            var scale = Math.sqrt(2 / fanIn);
            var W = [];
            var b = [];
            for (var j = 0; j < fanOut; j++) {
                var row = [];
                for (var i = 0; i < fanIn; i++) {
                    row.push(randn() * scale);
                }
                W.push(row);
                b.push(0);
            }
            this.weights.push(W);
            this.biases.push(b);
        }
    };

    Network.prototype.parameterCount = function () {
        var total = 0;
        for (var l = 1; l < this.layers.length; l++) {
            total += (this.layers[l - 1] + 1) * this.layers[l];
        }
        return total;
    };

    Network.prototype._activateHidden = function (z) {
        var a = [];
        for (var i = 0; i < z.length; i++) {
            a.push(leakyRelu(z[i]));
        }
        return a;
    };

    Network.prototype._activateOutput = function (z) {
        if (this.outputActivation === 'softmax') {
            return softmax(z);
        }
        return copyVector(z);
    };

    Network.prototype._forward = function (input) {
        if (!input || input.length !== this.layers[0]) {
            fail('predict() expected input of length ' + this.layers[0]);
        }
        var activations = [copyVector(input)];
        var zs = [];
        for (var l = 0; l < this.weights.length; l++) {
            var W = this.weights[l];
            var b = this.biases[l];
            var prev = activations[l];
            var z = [];
            for (var j = 0; j < W.length; j++) {
                var sum = b[j];
                for (var i = 0; i < prev.length; i++) {
                    sum += W[j][i] * prev[i];
                }
                z.push(sum);
            }
            zs.push(z);
            if (l === this.weights.length - 1) {
                activations.push(this._activateOutput(z));
            } else {
                activations.push(this._activateHidden(z));
            }
        }
        return { activations: activations, zs: zs };
    };

    Network.prototype.predict = function (input) {
        return this._forward(input).activations[this.layers.length - 1];
    };

    Network.prototype.inspect = function (input) {
        var forward = this._forward(input);
        var activations = [];
        var i;
        for (i = 0; i < forward.activations.length; i++) {
            activations.push(copyVector(forward.activations[i]));
        }
        return {
            layers: this.layers.slice(),
            activations: activations
        };
    };

    Network.prototype.getWeights = function () {
        var flat = [];
        for (var l = 0; l < this.weights.length; l++) {
            var W = this.weights[l];
            var b = this.biases[l];
            for (var j = 0; j < W.length; j++) {
                for (var i = 0; i < W[j].length; i++) {
                    flat.push(W[j][i]);
                }
                flat.push(b[j]);
            }
        }
        return flat;
    };

    Network.prototype.setWeights = function (flat) {
        var expected = this.parameterCount();
        if (!flat || flat.length !== expected) {
            fail('setWeights() expected ' + expected + ' values');
        }
        var index = 0;
        for (var l = 0; l < this.weights.length; l++) {
            var W = this.weights[l];
            var b = this.biases[l];
            for (var j = 0; j < W.length; j++) {
                for (var i = 0; i < W[j].length; i++) {
                    W[j][i] = flat[index++];
                }
                b[j] = flat[index++];
            }
        }
        return this;
    };

    Network.prototype.clone = function () {
        var copy = new Network({
            layers: this.layers,
            activation: this.activation,
            outputActivation: this.outputActivation
        });
        copy.setWeights(this.getWeights());
        return copy;
    };

    Network.prototype.mutate = function (sigma, rate) {
        if (typeof sigma !== 'number' || sigma < 0) {
            fail('mutate() requires sigma >= 0');
        }
        if (rate == null) {
            rate = 1;
        }
        if (typeof rate !== 'number' || rate < 0 || rate > 1) {
            fail('mutate() rate must be between 0 and 1');
        }
        if (sigma === 0 || rate === 0) {
            return this;
        }
        var l, j, i;
        for (l = 0; l < this.weights.length; l++) {
            for (j = 0; j < this.weights[l].length; j++) {
                for (i = 0; i < this.weights[l][j].length; i++) {
                    if (rate === 1 || Math.random() < rate) {
                        this.weights[l][j][i] += randn() * sigma;
                    }
                }
                if (rate === 1 || Math.random() < rate) {
                    this.biases[l][j] += randn() * sigma;
                }
            }
        }
        return this;
    };

    Network.prototype._outputDelta = function (activation, z, target, loss) {
        var n = activation.length;
        if (!target || target.length !== n) {
            fail('train() expected target of length ' + n);
        }
        var delta = [];
        var j;
        if (loss === 'ce' || loss === 'crossEntropy') {
            if (this.outputActivation !== 'softmax') {
                fail('cross-entropy loss requires outputActivation "softmax"');
            }
            for (j = 0; j < n; j++) {
                delta.push(activation[j] - target[j]);
            }
            return delta;
        }
        if (loss !== 'mse') {
            fail('unsupported loss "' + loss + '"');
        }
        if (this.outputActivation === 'softmax') {
            fail('mse is not supported with softmax; use loss "ce"');
        }
        for (j = 0; j < n; j++) {
            delta.push((2 / n) * (activation[j] - target[j]));
        }
        return delta;
    };

    Network.prototype._lossValue = function (activation, target, loss) {
        var n = activation.length;
        var j;
        if (loss === 'ce' || loss === 'crossEntropy') {
            var sum = 0;
            for (j = 0; j < n; j++) {
                var p = activation[j];
                if (p < 1e-12) {
                    p = 1e-12;
                }
                sum += -target[j] * Math.log(p);
            }
            return sum;
        }
        var mse = 0;
        for (j = 0; j < n; j++) {
            var err = activation[j] - target[j];
            mse += err * err;
        }
        return mse / n;
    };

    Network.prototype.train = function (input, target, options) {
        options = options || {};
        var learningRate = options.learningRate == null ? 0.01 : options.learningRate;
        var loss = options.loss || 'mse';
        if (typeof learningRate !== 'number' || learningRate <= 0) {
            fail('train() requires a positive learningRate');
        }

        var forward = this._forward(input);
        var activations = forward.activations;
        var zs = forward.zs;
        var last = activations.length - 1;
        var delta = this._outputDelta(activations[last], zs[zs.length - 1], target, loss);
        var value = this._lossValue(activations[last], target, loss);

        for (var l = this.weights.length - 1; l >= 0; l--) {
            var W = this.weights[l];
            var prev = activations[l];
            var nextDelta = [];
            var i;
            var j;
            if (l > 0) {
                for (i = 0; i < prev.length; i++) {
                    nextDelta[i] = 0;
                }
            }
            for (j = 0; j < W.length; j++) {
                for (i = 0; i < W[j].length; i++) {
                    if (l > 0) {
                        nextDelta[i] += W[j][i] * delta[j];
                    }
                    W[j][i] -= learningRate * delta[j] * prev[i];
                }
                this.biases[l][j] -= learningRate * delta[j];
            }
            if (l > 0) {
                var zPrev = zs[l - 1];
                for (i = 0; i < nextDelta.length; i++) {
                    nextDelta[i] *= leakyReluDerivative(zPrev[i]);
                }
                delta = nextDelta;
            }
        }

        return { loss: value };
    };

    return {
        create: function (options) {
            return new Network(options);
        },
        leakyAlpha: LEAKY_ALPHA
    };
}));
