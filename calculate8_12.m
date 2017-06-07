% Calculate the 9th to 12th slice of a given tensor
function X = calculate8_12(X)
  
	% calculate the 9th to 12th slice
    for j = 9:12
         % X = inter_production(1,2,3,4)

        % go through every pixel of a slice 
        for	x = 1:size(X,1)
        	for	y = 1:size(X,2)

        		% temp variable for the summary calculation
        		temp = 0;

        		% logical problem here
        		% code goes here
        		for	i = 1: size(X,1)
        			if i~=x && i~=y
        				temp = temp + inter_production(X(x,i,j-8,1),X(i,x,j-8,1),X(y,i,j-8,1),X(i,y,j-8,1));
        			end
        		end 

        		X(x,y,j,1) = temp;
        	end
        end

    % end of j loop
    end

% end of function  
end

% simple inter production function
function y = inter_production(a,b,c,d)
    y = a*b+c*d;
end
