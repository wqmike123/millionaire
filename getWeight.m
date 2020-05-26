function w = computeWeight(mu,cov,ndim)
%
target = 0.1;
stepSize = 0.01;
maxStep = 30000; 
acc = 1e-6*stepSize;
upper = 0.2;
nAsset = size(mu,1);
% compute subspace
[eVec,eVal]= eig(cov);
% compute lbd
%a = sqrt(mu' / cov * mu);
%lbd = a / target;   % could apply learning as well
% init w
w = eVec(:,1);
e = ones(nAsset,1);
l1 = 0.0; % lagrange multiplier to learn
l2 = 0.0;
for i = 1:maxStep,
    w_new = w - stepSize * (cov*w - l1*e - l2*mu);
    % project back
    temp = zeros(nAsset,1);
    for k = 1:ndim,
        temp = temp + w_new'*eVec(:,k) / (eVec(:,k)'*eVec(:,k)) * eVec(:,k);     
    end
    temp(temp<0) = 0.0;
    temp(temp>upper) = upper;
    %temp = temp / sum(temp);
    maxChange = max(abs(temp - w));    
    w = temp;
    deltal1 =  - stepSize * (w'*e - 1);
    l1 = l1 - deltal1;
    deltal2 = - stepSize * (w'*mu - target);
    l2 = l2 - deltal2;
    if (maxChange < acc) && (max(abs(deltal1))<acc) && (max(deltal2)<acc),
        break;
    end
end
end
