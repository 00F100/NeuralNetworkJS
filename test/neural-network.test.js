'use strict';

var test = require('node:test');
var assert = require('node:assert/strict');
var NeuralNetwork = require('../src/neural-network.js');

function meanLoss(net, samples, loss) {
    var total = 0;
    for (var i = 0; i < samples.length; i++) {
        var out = net.predict(samples[i].input);
        if (loss === 'ce') {
            var p = out[samples[i].target.indexOf(1)];
            if (p < 1e-12) {
                p = 1e-12;
            }
            total += -Math.log(p);
        } else {
            var se = 0;
            for (var j = 0; j < out.length; j++) {
                var err = out[j] - samples[i].target[j];
                se += err * err;
            }
            total += se / out.length;
        }
    }
    return total / samples.length;
}

test('create requires layer sizes', function () {
    assert.throws(function () {
        NeuralNetwork.create({});
    });
    assert.throws(function () {
        NeuralNetwork.create({ layers: [3] });
    });
});

test('inspect returns one activation vector per layer', function () {
    var net = NeuralNetwork.create({ layers: [6, 8, 8, 3] });
    var input = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
    var view = net.inspect(input);
    assert.deepEqual(view.layers, [6, 8, 8, 3]);
    assert.equal(view.activations.length, 4);
    assert.deepEqual(view.activations[0], input);
    assert.equal(view.activations[1].length, 8);
    assert.equal(view.activations[2].length, 8);
    assert.equal(view.activations[3].length, 3);
    assert.deepEqual(view.activations[3], net.predict(input));
});

test('predict returns one value per output unit', function () {
    var net = NeuralNetwork.create({ layers: [6, 8, 8, 3] });
    var out = net.predict([0.1, 0.2, 0.3, 0.4, 0.5, 0]);
    assert.equal(out.length, 3);
    for (var i = 0; i < out.length; i++) {
        assert.equal(typeof out[i], 'number');
        assert.equal(Number.isFinite(out[i]), true);
    }
});

test('parameter count is sum of (in + 1) * out per layer', function () {
    var net = NeuralNetwork.create({ layers: [6, 8, 8, 3] });
    var expected = (6 + 1) * 8 + (8 + 1) * 8 + (8 + 1) * 3;
    assert.equal(net.parameterCount(), expected);
    assert.equal(net.getWeights().length, expected);
});

test('getWeights / setWeights round-trip', function () {
    var net = NeuralNetwork.create({ layers: [2, 3, 1] });
    var original = net.getWeights();
    var copy = original.slice();
    for (var i = 0; i < copy.length; i++) {
        copy[i] = i * 0.01;
    }
    net.setWeights(copy);
    assert.deepEqual(net.getWeights(), copy);
    net.setWeights(original);
    assert.deepEqual(net.getWeights(), original);
});

test('setWeights rejects the wrong length', function () {
    var net = NeuralNetwork.create({ layers: [2, 2] });
    assert.throws(function () {
        net.setWeights([1, 2, 3]);
    });
});

test('clone is independent of the source network', function () {
    var net = NeuralNetwork.create({ layers: [2, 4, 1] });
    var cloned = net.clone();
    assert.deepEqual(cloned.getWeights(), net.getWeights());
    assert.deepEqual(cloned.layers, net.layers);
    cloned.mutate(0.5);
    assert.notDeepEqual(cloned.getWeights(), net.getWeights());
});

test('mutate(0) leaves weights unchanged and mutate(sigma) changes them', function () {
    var net = NeuralNetwork.create({ layers: [2, 3, 1] });
    var before = net.getWeights();
    net.mutate(0);
    assert.deepEqual(net.getWeights(), before);
    net.mutate(0.2, 0);
    assert.deepEqual(net.getWeights(), before);
    net.mutate(0.2);
    assert.notDeepEqual(net.getWeights(), before);
});

test('softmax outputs sum to 1', function () {
    var net = NeuralNetwork.create({
        layers: [4, 5, 3],
        outputActivation: 'softmax'
    });
    var out = net.predict([0.2, -0.1, 1.5, 0.0]);
    assert.equal(out.length, 3);
    var sum = 0;
    for (var i = 0; i < out.length; i++) {
        assert.ok(out[i] >= 0);
        assert.ok(out[i] <= 1);
        sum += out[i];
    }
    assert.ok(Math.abs(sum - 1) < 1e-9);
});

test('API does not read game or melonJS globals', function () {
    global.game = { ia: { weightSeq: [1, 2, 3] } };
    global.me = { state: {} };
    var net = NeuralNetwork.create({ layers: [2, 2, 1] });
    var out = net.predict([0.5, -0.5]);
    assert.equal(out.length, 1);
    assert.equal(Number.isFinite(out[0]), true);
    delete global.game;
    delete global.me;
});

test('train() on XOR reduces mean squared error', function () {
    var samples = [
        { input: [0, 0], target: [0] },
        { input: [0, 1], target: [1] },
        { input: [1, 0], target: [1] },
        { input: [1, 1], target: [0] }
    ];
    var net = NeuralNetwork.create({
        layers: [2, 8, 1],
        activation: 'leakyRelu',
        outputActivation: 'linear'
    });
    var before = meanLoss(net, samples, 'mse');
    for (var epoch = 0; epoch < 4000; epoch++) {
        for (var i = 0; i < samples.length; i++) {
            var sample = samples[(i + epoch) % samples.length];
            net.train(sample.input, sample.target, { learningRate: 0.08, loss: 'mse' });
        }
    }
    var after = meanLoss(net, samples, 'mse');
    assert.ok(after < before, 'expected loss to drop from ' + before + ' to ' + after);
    assert.ok(after < 0.05, 'expected XOR mse below 0.05, got ' + after);
});

test('train() with softmax and cross-entropy reduces loss', function () {
    var samples = [
        { input: [1, 0], target: [1, 0] },
        { input: [0, 1], target: [0, 1] }
    ];
    var net = NeuralNetwork.create({
        layers: [2, 6, 2],
        outputActivation: 'softmax'
    });
    var before = meanLoss(net, samples, 'ce');
    for (var epoch = 0; epoch < 800; epoch++) {
        for (var i = 0; i < samples.length; i++) {
            net.train(samples[i].input, samples[i].target, { learningRate: 0.2, loss: 'ce' });
        }
    }
    var after = meanLoss(net, samples, 'ce');
    assert.ok(after < before, 'expected CE loss to drop from ' + before + ' to ' + after);
    var a = net.predict([1, 0]);
    var b = net.predict([0, 1]);
    assert.ok(a[0] > a[1], 'expected class 0 for [1,0], got ' + a);
    assert.ok(b[1] > b[0], 'expected class 1 for [0,1], got ' + b);
});
