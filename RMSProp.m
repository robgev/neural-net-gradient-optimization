syms x1 x2;
f = -2 * exp(-((x1 - 1)^2 + x2^2) / .2) + -3 * exp(-((x1 + 1)^2 + x2^2) / .2) + x1^2 + x2^2;
alpha = 1e-2;
b = 0.9;
e = 1e-8;
params = [x1, x2];
testVals1 = [0.3, -1.8];
testVals2 = [0.79, -1.7];
testVals3 = [-1, -1.6];

%RMSProp_grad(f, params, testVals1, alpha, b, e);
%RMSProp_grad(f, params, testVals2, alpha, b, e);
RMSProp_grad(f, params, testVals3, alpha, b, e);

function result = RMSProp_grad(f, params, weights, alpha, b, e)
    disp('Initial values are:');
    disp(weights);
    n = 0;
    result = weights;
    grads = gradient(f, params);
    s = 0;
    ep = 1e-5;
    while norm(double(subs(grads, params, result))) >= ep
            n = n + 1;
            grad_vals = double(subs(grads, params, result))';
            % the only part different from 
            % the previous algo is here
            s = (b * s) + ((1 - b) * grad_vals .^ 2);
            % in case you want to do bias correction
            % s_corr = s / (1 - b ^ n);
            step = grad_vals ./ (sqrt(s) + e); % sqrt(s_corr)
            result = result - (alpha * step);
    end
    disp(['Number of steps: ', num2str(n)]); 
    disp('Calculated Weights are:');
    disp(result);
    disp('');
end



