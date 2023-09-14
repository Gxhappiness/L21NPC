function W = L21NPC(DATA,delta1,delta2,x0,itmax,epsmax,zeta) 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % % % L21NPC: 
% L2-L1 non-parallel classifier
% the number of category:k
% Input:
%   DATA:type is (k+2)*1 cell,
%        DATA{i}=the i-th trainset i<=k
%        DATA{k+1},val;DATA{k+2},test
%   delta1: Weight of Goal 2 relative to Goal 1 term parameter.
%   delta2: regularization term parameter.
%   w0: Initial hyperplane direction and bias
%   itmax: Maximun iteration number
%   epsmax: Tolerance
%   zeta: zeta in affinity


% Output:
%   W:the matrix of hyperplane direction and bias ((nfeature+1)*k)




% % % % Eample:
% Atrain = rand(30,2);
% Btrain = rand(30,2) + 1;
% DATA=cell(4,1);
% DATA{1}=Atrain;
% DATA{2}=Btrain;
% x0 = ones(size(Atrain,2) + 1,1); % Initialization
% delta1 = 1; 
% delta2 = 0.1; 
% itmax=100;
% epsmax=0.02;
% zeta=1;
% W = L21NPC(DATA,delta1,delta2,x0,itmax,epsmax,zeta) ;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initailization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


k=length(DATA)-2;
[~, nfea]=size(DATA{1}); % nfeature
W=zeros(nfea+1,k);%the matrix of hyperplane direction and bias ((nfeature+1)*k)
ntrain=zeros(k,1);%the number of train point
I=eye(nfea+1);%Identity matrix
sd=zeros(k,k);  %similarity distance matrix
aff=zeros(k,k);%affinity matrix
eaff=zeros(k,k); %exp(1-aff(i,j)) matrix
pro=zeros(k,k); %proportion matrix(softmax)
sump=zeros(k,1);%sum of the proportion

for i=1:k-1
    for j=i+1:k
        sd(i,j)=norm((DATA{i}'*DATA{i}-DATA{j}'*DATA{j}),1)/nfea;
    end
end
sd=sd+sd';

for i=1:k-1
    for j=i+1:k
        aff(i,j)=zeta/(sd(i,j)+miu);
    end
end
aff=aff+aff';

for i=1:k-1
    for j=i+1:k
         eaff(i,j)=exp(1-aff(i,j));
    end
end
eaff=eaff+eaff';

for i=1:k
    sump(i)=sum(eaff(i,:));
end

for i=1:k
    ntrain(i)=size(DATA{i},1);
end

for i=1:k-1
    for j=i+1:k
           pro(i,j)=eaff(i,j)/sump(i);
    end
end
pro=pro+pro';



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% start method!!!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


for i=1:k
    Gi=(1-delta1)*[DATA{i},ones(ntrain(i),1)]'*[DATA{i},ones(ntrain(i),1)]+delta2*I;
    Ri=chol(Gi);
    xj=x0;
    iter=0;
    while(iter<itmax)
        xj1=xj;
        xj=delta1*Ri \ (Ri' \ T(xj1,nfea,i,pro,DATA,k,ntrain));
        if norm(xj-xj1)<epsmax
            break;
        end
        iter=iter+1;
    end
    W(:,i)=xj/norm(xj);
end
end
