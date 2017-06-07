function magicsquare(n)
if ischar(n)
    n = str2num(n);
 end
disp(magic(n))