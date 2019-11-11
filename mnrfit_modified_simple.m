%% This function is to estimate the parameters by Newton Raphson for given latent ratio omega_0
%%%% Input: 
%        x: covariates
%        y: binary outcome
%        omega_0: latent ratio
%%%% Output:
%        b: estimated coefficients 
function b = mnrfit_modified_simple(x,y,omega_0)
b = zeros(1,size(x,2)+1)';
iter=0;
iterLim=100;
while iter <= iterLim
    b_old=b;
    iter=iter+1;
    pihat = 1-1./(exp([ones(size(x,1),1),x]*b_old)+1);
    W = sparse(1:length(y),1:length(y),pihat,length(y),length(y))*sparse(1:length(y),1:length(y),1-pihat,length(y),length(y));
    Wz = W*([ones(size(x,1),1),x]*b_old)+y'+omega_0*(1-y')-pihat;
    XWX = [ones(size(x,1),1),x]'*W*[ones(size(x,1),1),x];
    XWZ = [ones(size(x,1),1),x]'*Wz;
    b = XWX \ XWZ;
    
    cvgTest = abs(b-b_old)>1e-6;
    
    if iter>iterLim || ~any(cvgTest(:))
        break;
    end
    
end
end