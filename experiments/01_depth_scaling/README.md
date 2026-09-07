# Depth scaling

How increasing number of layers impacts the model?

## Fixed parameters

- B = 2 — batch size
- T = 128 — sequence length used in the benchmark
- T_max = 256 — maximum context length supported by the model
- D = 128 — embedding dimension / model width
- H = 4 — number of attention heads
- V = 50257 — vocabulary size
- Precision: FP32
- Attention: naive

## Changing parameter:

- L = 1, 2, 4, 8, 12, 16 (number of layers, blocks)

## Questions:

- [] How does increasing Transformer depth affect predicted FLOPs and measured runtime?

- [] What fraction of the predicted forward FLOPs comes from the Transformer blocks?

- [] How does increasing Transformer depth affect memory consumption?

- [] When the Transformer blocks start to dominate in FLOPs?

## Predictions

### Forward pass

We approximate the dominant matrix multiplications, counting one multiply-add as two FLOPs:

$$
F_{\text{block}} \approx 24BTD^2 + 4BT^2D
$$

$$
F_{\text{head}} \approx 2BTDV
$$

$$
F_{\text{forward}}(L) \approx L F_{\text{block}} + F_{\text{head}}
$$

For B=2, T=128, D=128 and V=50257:

$$
F_{\text{block}} \approx 117{,}440{,}512 \text{ FLOPs}
\approx 0.1174 \text{ GFLOPs}
$$

$$
F_{\text{head}} \approx 3{,}293{,}642{,}752 \text{ FLOPs}
\approx 3.2936 \text{ GFLOPs}
$$

### When do blocks match the head?

$$
L_* F_{\text{block}} = F_{\text{head}}
$$

$$
L_* =
\frac{2BTDV}{24BTD^2+4BT^2D}
=
\frac{V}{12D+2T}
$$

$$
L_* = \frac{50257}{12\cdot128+2\cdot128}
\approx 28.05
$$

### Backward pass

For a matrix multiplication, backward computes gradients for both inputs:

$$
Y=XW,\qquad
\nabla_X=\nabla_Y W^\top,\qquad
\nabla_W=X^\top\nabla_Y
$$

Each gradient multiplication has the same FLOP count as the forward multiplication. Using this approximation throughout the dominant matmuls:

$$
F_{\text{backward}} \approx 2F_{\text{forward}}
$$

$$
F_{\text{forward+backward}} \approx 3F_{\text{forward}}
$$

For L=2:

$$
F_{\text{forward}} \approx
3{,}528{,}523{,}776 \text{ FLOPs}
\approx 3.5285 \text{ GFLOPs}
$$

$$
F_{\text{backward}} \approx
7{,}057{,}047{,}552 \text{ FLOPs}
\approx 7.0570 \text{ GFLOPs}
$$

$$
F_{\text{forward+backward}} \approx
10{,}585{,}571{,}328 \text{ FLOPs}
\approx 10.5856 \text{ GFLOPs}
$$

This is a matmul approximation, not an exact count of every backward operation.

### Full optimizer step

A full training step includes forward, backward and an AdamW update:

$$
F_{\text{step}} \approx 3F_{\text{forward}} + F_{\text{AdamW}}
$$

AdamW uses the gradients to update the weights. It also maintains
two moving averages for each parameter: the gradient and its square.

