f = @(x) -2 .* exp(-((x(1) - 1).^2 + x(2).^2) ./ .2) + -3 .* exp(-((x(1) + 1).^2 + x(2).^2) ./ .2) + x(1).^2 + x(2).^2;
fminsearch(f, [-2, 0])
fminsearch(f, [0.5, 1])