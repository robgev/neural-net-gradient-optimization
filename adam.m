syms x1 x2;
f = -2 * exp(-((x1 - 1)^2 + x2^2) / .2) + -3 * exp(-((x1 + 1)^2 + x2^2) / .2) + x1^2 + x2^2;
alpha = 1e-2;
b1 = 0.9;
b2 = 0.999;
e = 1e-8;
params = [x1, x2];
testVals1 = [0.3, -1.8];

weights = adam_gr(f, params, testVals1, alpha, b1, b2, e);
disp('Calculated Weights are:'); 
disp(weights);

function result = adam_gr(f, params, weights, alpha, b1, b2, e)
    n = 0;
    result = weights;
    grads = gradient(f, params);
    v = 0;
    s = 0;
    ep = 1e-5;
    while norm(double(subs(grads, params, result))) >= ep
            n = n + 1;
            grad_vals = double(subs(grads, params, result))';
            % we calculate v as in momentum
            v = (b1 * v) + ((1 - b1) * grad_vals);
            % and s as in RMSProp
            s = (b2 * s) + ((1 - b2) * grad_vals .^ 2);
            % bias corrections
            v_corr = v / (1 - b1 ^ n);
            s_corr = s / (1 - b2 ^ n);
            % now we use both in a combined step
            step = v_corr ./ (sqrt(s_corr) + e);
            % and then update weights
            result = result - (alpha * step);
    end
    disp(['Number of steps: ', num2str(n)]); 
end



