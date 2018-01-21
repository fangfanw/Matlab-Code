clear; clc; close all; 

load SimulatedData.mat;
[J, nSamples] = size(Yt);
t = 1:nSamples;  
covariates = [ones(1,nSamples); cos(2*pi*t./5); sin(2*pi*t./5)];  
D0 = [1 0.3 -0.05; 
      2 0.1 -0.03; 
      3 0.5 -0.07; 
      4 0.4 -0.01;
      5 0.6 -0.1];
[J,I] = size(D0);   
A = ones(5,1); 
d3 = [1/(.16^2); 1/(.37^2); 1/(.294^2); 1/(.219^2); 1/(.226^2)]; 


DEst  = PoisPara_fminunc(Yt, covariates);  
EstTrend = exp(DEst*covariates);  
etahat = Yt./EstTrend;              
etabar = mean(etahat,2);
    

Omega1jhat = cell(1,J); Omegayjhat = cell(1,J);
SqResid = (Yt - EstTrend).^2; 
for j = 1:J
    EstXtTrend0 = covariates.*repmat(EstTrend(j,:), I, 1); 
    Omega1jhat{j} = nSamples^(-1).*(EstXtTrend0*covariates'); 
    EstXtResid = covariates.*repmat(SqResid(j,:), I, 1); 
    Omegayjhat{j} = nSamples^(-1).*(EstXtResid*covariates'); 
end
    
SigmaEtaknonneg = zeros(J,J,nSamples);
SigmaEtakneg = zeros(J,J,nSamples-1);    
for k = 0:(nSamples-1)
    etahatbiask = etahat(:,(k+1):nSamples) - repmat(etabar, 1, nSamples-k);         
    etahatbias0 = etahat(:,1:(nSamples-k)) - repmat(etabar, 1, nSamples-k);
    SigmaEtaknonneg(:,:,k+1) = etahatbiask*etahatbias0'./(nSamples); 
end
for k = 1:(nSamples-1)
    etahatbiask = etahat(:,1:(nSamples-k)) - repmat(etabar, 1, nSamples-k);
    etahatbias0 = etahat(:,(k+1):nSamples) - repmat(etabar, 1, nSamples-k); 
    SigmaEtakneg(:,:,k) = etahatbiask*etahatbias0'./(nSamples); 
end   

Sigmahat = cell(J,J); 
for j = 1:J
    EstXtTrendj = covariates.*repmat(EstTrend(j,:), I, 1); 
    for h = 1:J 
        EstXtTrendh = covariates.*repmat(EstTrend(h,:), I, 1); 
        Crossprodnonnegjh = zeros(I,I,nSamples); 
        Crossprodnegjh = zeros(I,I,nSamples-1);              
        for k = 0:(nSamples-1) 
            tempjk = EstXtTrendj(:,k+1:nSamples); 
            temph0 = EstXtTrendh(:,1:nSamples-k); 
            Whatk = nSamples^(-1).*(tempjk*temph0');  
            Crossprodnonnegjh(:,:,k+1) = Whatk.*SigmaEtaknonneg(j,h,k+1); 
        end
        for k = 1:(nSamples-1) 
            tempjk = EstXtTrendj(:,1:nSamples-k);
            temph0 = EstXtTrendh(:,k+1:nSamples); 
            Whatk = nSamples^(-1).*(tempjk*temph0');             
            Crossprodnegjh(:,:,k) = Whatk.*SigmaEtakneg(j,h,k); 
        end            
        Crossprodposjh = Crossprodnonnegjh(:,:,2:nSamples);
        if h == j
            Sigmahat{j,h} = Omegayjhat{1,j}-Omega1jhat{1,j}+sum(Crossprodposjh,3)+sum(Crossprodnegjh,3); 
        else
            Sigmahat{j,h} = sum(Crossprodnonnegjh,3)+sum(Crossprodnegjh,3);                  
        end
    end            
end
SigmahatMat = cell2mat(Sigmahat); 
Omega1hat = blkdiag(Omega1jhat{1,:}); 
[VSigma, DSigma] = eig(SigmahatMat); 
DSigma(DSigma<0) = 0; 
SigmahatMatnonneg = VSigma*DSigma*VSigma'; 
Omega23 = diag(Omega1hat\SigmahatMatnonneg/Omega1hat); 
AVarEstb =  diag(inv(Omega1hat)) + Omega23; 
StdErrb = sqrt(AVarEstb)/sqrt(nSamples); 


Cn = log(nSamples)*nSamples^(-1);
k0 = 20; 
SigmaEtak2 = zeros(J,J,k0); 
for k = 1:k0
    etahatbias0 = etahat(:,1:(nSamples-k)) - repmat(etabar, 1, nSamples-k);
    etahatbias1 = etahat(:,(k+1):nSamples) - repmat(etabar, 1, nSamples-k); 
    crossprod = etahatbias0*etahatbias1'; 
    SigmaEtak = crossprod./nSamples;  
    SigmaEtak2(:,:,k) = SigmaEtak*SigmaEtak'; 
end

Lhat = sum(SigmaEtak2,3);
[V,D] = eig(Lhat);       
eigvalsL = ones(1,J)*D;  
eigvalsL = sort(eigvalsL,'descend');   
plot(eigvalsL, '*','MarkerSize', 3); 
xlim([1,J]); ylim([0,0.3]) 
title(['k_0=' num2str(k0)])  

eigvalsLnoise = eigvalsL + repmat(Cn,1,J);  
eigvalsLnoise2= eigvalsLnoise(2:J);
eigvalsLnoise1= eigvalsLnoise(1:J-1);
ratios = eigvalsLnoise2./eigvalsLnoise1;
find(ratios == min(ratios))


