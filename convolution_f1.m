% --- convolution 2d ---
% mention <m> is from 1 to 28-5+1:
% this means that this conv is like:
%              x x x x x                o o o x x     
%              x x x x x                o o o x x      m x x 
% primary img: x x x x x  conv kernel:  o o o x x ->   x x x  ...
%              x x x x x                x x x x x      x x x 
%              x x x x x                x x x x x     
%
% but not like: then m from 1 to 28
%                                      o o o
%              x x x x x               o o o x x x    m x x x x 
%              x x x x x               o o o x x x    x x x x x
% primary img: x x x x x  conv kernel:   x x x x x -> x x x x x ...
%              x x x x x                 x x x x x    x x x x x
%              x x x x x                 x x x x x    x x x x x


% --- computing convolution results ---
function [state] = convolution(data, kernel)
[data_row,data_col] = size(data);
[kernel_row,kernel_col] = size(kernel);
for m=1:data_col-kernel_col+1
    for n=1:data_row-kernel_row+1
        state(m,n) = sum(sum(data(m:m+kernel_row-1, n:n+kernel_col-1).*kernel));
    end
end
end

