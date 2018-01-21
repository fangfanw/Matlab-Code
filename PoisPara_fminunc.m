function ParametersEst = PoisPara_fminunc(Yt, covariates, options)
if (nargin <= 2) || isempty(options)
    options  =  optimset('fminunc');
    options  =  optimset(options , 'TolFun'      , 1e-004);
    options  =  optimset(options , 'Display'     , 'off');
    options  =  optimset(options , 'Diagnostics' , 'off');
    options  =  optimset(options , 'LargeScale'  , 'on');
    options  =  optimset(options , 'MaxFunEvals' , 400*(3));
    options  =  optimset(options , 'GradObj'     , 'on');
    options  =  optimset(options , 'Hessian'     , 'on');
end

J = size(Yt,1); 
I = size(covariates,1); 
ParametersEst = zeros(J, I); 
for j = 1:J
   Ytj = Yt(j,:)'; 
    D0j = (covariates*covariates')^(-1)*(covariates*log(Ytj+1)); 
    [D1, likelihood, EXITFLAG] =  fminunc('ObjLn', D0j, options, Ytj, covariates);
    ParametersEst(j,:) = D1'; 
    if EXITFLAG<=0
        fprintf(1,'Not Sucessful! \n')
    end
end
