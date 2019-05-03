syms x1 x2;
f = -2 * exp(-((x1 - 1)^2 + x2^2) / .2) + -3 * exp(-((x1 + 1)^2 + x2^2) / .2) + x1^2 + x2^2;
% Global min of this function approx. [-0.93, 0]
% Another local min of this function approx. [0.93, 0]
alpha = 1e-2;
b = 0.9;
params = [x1, x2];
testVals1 = [0.3, -1.8];
testVals2 = [0.95, 1.55];

disp('Initial values are:');
disp(testVals1);
weights = momentum_gr(f, params, testVals1, alpha, b);
disp('Calculated Weights are:'); 
disp(weights);
disp('Initial values are:');
disp(testVals2);
weights = momentum_gr(f, params, testVals2, alpha, b);
disp('Calculated Weights are:'); 
disp(weights);

function result = momentum_gr(f, params, weights, alpha, b)
    % f is the concerned function
    % params are symbolic variables
    % weights are values vector
    % we will use termination condition
    % when gradient is smaller than ep
    % we stop iterating
    n = 0; % step counter
    % we want to return the final weights
    % but do not want to mutate the inputs
    result = weights; 
    % calculating the symbolic gradient
    % w.r.t params
    grads = gradient(f, params);
    % we do not have initial "velocity"
    v = 0;
    ep = 1e-5;
    while norm(double(subs(grads, params, result))) >= ep
            n = n + 1;
            % we evaluate the value of
            % the gradient using subs to substitute
            % params with result vector's values in 
            % gradient vector
            % as it's a symbolic value, we want to turn
            % it back to double value, hence the double
            % ' at the end takes the transpose
            % as the result is a column vector but we need
            % a row vector
            grad_vals = double(subs(grads, params, result))';
            % calculating the momentum
            v = (b * v) + ((1 - b) * grad_vals);
            % performing the actual update;
            result = result - (alpha * v);
    end
    % just to display number of steps
    disp(['Number of steps: ', num2str(n)]); 
end



