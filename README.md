# NeuralNetworkJS

Multilayer perceptron in a single JavaScript file. No runtime dependencies. Works in the browser (script tag) and in Node (`require`).

The library does not know about any game. It only maps numeric vectors to numeric vectors, stores its own weights, and optionally updates them with SGD. A consumer (for example ET Bilu) creates a network, calls `predict`, and — if it uses evolution — `clone` / `mutate` / `getWeights`.

## Install

Copy `src/neural-network.js` or load it as a script:

```html
<script src="vendor/neural-network.js"></script>
```

```javascript
var NeuralNetwork = require('./src/neural-network.js');
```

## Architecture

Fully connected MLP with bias on every hidden and output unit:

```
input  →  Dense + leaky ReLU  →  …  →  Dense + linear|softmax  →  output
```

For `layers: [n0, n1, …, nL]` layer `k` has `nk` units. Weights from layer `k-1` to `k` are a matrix `W^(k)` of shape `(nk, n_{k-1})` plus a bias `b^(k)` of length `nk`.

Forward, hidden layers (`α = 0.01`):

```
z^(k) = W^(k) a^(k-1) + b^(k)
a^(k) = leakyReLU(z^(k)) = z if z > 0 else α z
```

Output:

- `linear`: `a = z` (logits; a consumer can `argmax`)
- `softmax`: `a_i = exp(z_i - max z) / Σ exp(z_j - max z)`

Weights are initialized with **He / Kaiming**: `W ~ N(0, √(2 / fan_in))`, biases `0`. That keeps leaky ReLU activations from dying at start.

Parameter count:

```
Σ_k (n_{k-1} + 1) * n_k
```

(`+1` is the bias.) `getWeights()` serializes each layer as rows of `W` followed by the matching bias.

## Training (optional)

`train(input, target, { learningRate, loss })` is one SGD step:

1. Forward, cache `z` and `a`.
2. Output error:
   - MSE + linear: `δ = (2 / n) (a - y)`
   - Cross-entropy + softmax: `δ = a - y` (`y` one-hot)
3. Backprop: `δ^(k) = (W^(k+1)ᵀ δ^(k+1)) ⊙ leakyReLU'(z^(k))`
4. Update: `W ← W - η δ aᵀ`, `b ← b - η δ`

The game does **not** have to call `train`. Neuroevolution only needs `predict`, `clone`, and `mutate`.

`mutate(sigma, rate)` adds `N(0, σ)` to weights and biases. `rate` (default `1`) is the chance each parameter is touched. That is generic noise, not game logic.

## API

### `NeuralNetwork.create(options) → net`

| option | default | meaning |
|---|---|---|
| `layers` | required | `[input, hidden…, output]` positive integers |
| `activation` | `'leakyRelu'` | hidden activation |
| `outputActivation` | `'linear'` | `'linear'` or `'softmax'` |

### Instance

```javascript
var net = NeuralNetwork.create({
  layers: [6, 8, 8, 3],
  activation: 'leakyRelu',
  outputActivation: 'linear'
});

net.predict([0.1, 0.2, 0.3, 0.4, 0.5, 0]); // number[]
net.parameterCount();                       // integer
net.getWeights();                           // number[]
net.setWeights(array);                      // chainable
net.clone();                                // independent copy
net.mutate(0.06, 0.3);                      // Gaussian noise on ~30% of params, chainable
net.train(input, target, {
  learningRate: 0.01,
  loss: 'mse'        // or 'ce' / 'crossEntropy' with softmax
});                 // { loss: number }
```

## Supervised example (XOR)

```javascript
var net = NeuralNetwork.create({ layers: [2, 8, 1] });
var data = [
  { x: [0, 0], y: [0] },
  { x: [0, 1], y: [1] },
  { x: [1, 0], y: [1] },
  { x: [1, 1], y: [0] }
];
for (var epoch = 0; epoch < 4000; epoch++) {
  for (var i = 0; i < data.length; i++) {
    net.train(data[i].x, data[i].y, { learningRate: 0.08, loss: 'mse' });
  }
}
```

## Neuroevolution example

A host program owns selection. The network only evaluates and perturbs parameters:

```javascript
var parent = NeuralNetwork.create({ layers: [6, 8, 8, 3] });
var child = parent.clone().mutate(0.06, 0.3);
var logits = child.predict(features);
var action = 0;
for (var i = 1; i < logits.length; i++) {
  if (logits[i] > logits[action]) action = i;
}
```

ET Bilu uses this pattern (features from the nearest obstacle, actions jump / duck / idle). The library has no import of that game.

## Tests (Docker)

No local Node or npm is required:

```bash
docker compose run --rm test
```

The image runs `node --test` against `test/`. Coverage includes XOR loss dropping, softmax summing to 1, weight round-trip, clone isolation, mutate, parameter count, and that the API does not read `game` / `me` globals.
