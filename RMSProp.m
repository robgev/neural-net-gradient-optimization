function result = RMSProp(f, params, weights, alpha, b, e)
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
