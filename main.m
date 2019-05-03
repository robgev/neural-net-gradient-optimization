syms x1 x2;
f = -2 * exp(-((x1 - 1)^2 + x2^2) / .2) + -3 * exp(-((x1 + 1)^2 + x2^2) / .2) + x1^2 + x2^2;
alpha = 1e-2;
b1 = 0.9;
b2 = 0.999;
e = 1e-8;
params = [x1, x2];
testVals1 = [0.3, -1.8];
testVals2 = [0.79, -1.7];
testVals3 = [-1, -1.6];

momentum(f, params, testVals1, alpha, b1);
RMSProp_grad(f, params, testVals1, alpha, b1, e);
adam(f, params, testVals1, alpha, b1, b2, e);

momentum(f, params, testVals2, alpha, b1);
RMSProp_grad(f, params, testVals2, alpha, b1, e);
adam(f, params, testVals1, alpha, b1, b2, e);

momentum(f, params, testVals2, alpha, b1);
RMSProp_grad(f, params, testVals3, alpha, b1, e);
adam(f, params, testVals1, alpha, b1, b2, e);