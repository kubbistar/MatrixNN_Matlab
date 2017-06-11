%% -------------------------------------
% the size of node
sizes = zeros(25,1);
sizes(:) = 100;
% the edge info 
edges = zeros(1,3);

% counting for how many edges
n = 1;

for	i = 1:size(B1,1)
	for	j = 1:size(B1,2)
		if B1(i,j) ~= 0
			sizes(i,1) = sizes(i,1) + abs(B1(i,j));
			sizes(j,1) = sizes(j,1) + abs(B1(i,j));

			edges(n,1) = i;
			edges(n,2) = j;
			edges(n,3) = B1(i,j);
			n = n+1;
		end
	end
end
csvwrite('size_B1.csv',sizes);
csvwrite('edge_B1.csv',edges);


%% -------------------------------------
% the size of node
sizes = zeros(25,1);
sizes(:) = 100;
% the edge info 
edges = zeros(1,3);

% counting for how many edges
n = 1;

for	i = 1:size(B2,1)
	for	j = 1:size(B2,2)
		if B2(i,j) ~= 0
			sizes(i,1) = sizes(i,1) + abs(B2(i,j));
			sizes(j,1) = sizes(j,1) + abs(B2(i,j));
			
			edges(n,1) = i;
			edges(n,2) = j;
			edges(n,3) = B2(i,j);
			n = n+1;
		end
	end
end
csvwrite('size_B2.csv',sizes);
csvwrite('edge_B2.csv',edges);