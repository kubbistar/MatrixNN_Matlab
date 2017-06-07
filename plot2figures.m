% c_m: your loss output
% c_g: Gao's loss output
% c_h: Hoff's loss output

function plot2figures(c_m, c_g_1, c_g_2, c_g_3, c_g_4, c_h)
	x = 1:length(c_m);

	c_g = (c_g_4+c_g_3+c_g_2+c_g_1)/4;

	% plot c_m against c_g
	figure
	plot(x, c_m, '-r', x, c_g,'-b','Linewidth',2);
	legend('TensorNet','MatNet');
	% plot c_m against c_h
	figure
	plot(x, c_m, '-r', x, c_h,'-g','Linewidth',2);
    legend('TensorNet','MLTR');
end

