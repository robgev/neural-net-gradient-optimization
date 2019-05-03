% f = @(x) -2 .* exp(-((x(1) - 1).^2 + x(2).^2)) + -3 .* exp(-((x(1) + 1).^2 + x(2).^2)) + x(1).^2 + x(2).^2;
f = @(x1, x2) -2 .* exp(-((x1 - 1).^2 + x2.^2) ./ .2) + -3 .* exp(-((x1 + 1).^2 + x2.^2) ./ .2) + x1.^2 + x2.^2;
[X, Y] = meshgrid(-2:0.001:2);
z = f(X,Y);
%fminsearch(f, [0.5, -1]);
%fminsearch(f, [0.5, 1]);
contourf(X,Y,z,10)