function [Lnj, Gradj, Hessj] = ObjLn(Dj, Ytj, covariates)

Dj = Dj(:); Ytj = Ytj(:); 
[nSamples, J] = size(Ytj);
if J ~= 1
        error('Data must be a row vector.')
end   

DXt = Dj'*covariates; 
eDXt = exp(DXt);
I = size(covariates,1); 

llt = Ytj'.*DXt - eDXt;  
Lnj = (-1)*sum(llt); 

if nargout > 1  
    dllt = zeros(I,nSamples);
    for t = 1:nSamples
    dllt(:,t) = covariates(:,t)*Ytj(t) - covariates(:,t)*exp(Dj'*covariates(:,t));         
    end
    Gradj = (-1)*sum(dllt,2); 
end

if nargout > 2  
    Hllt = zeros(I,I,nSamples);
    for t = 1:nSamples
    Hllt(:,:,t) =  - covariates(:,t)*exp(Dj'*covariates(:,t))*(covariates(:,t)');          
    end
    Hessj = (-1)*sum(Hllt,3); 
end

