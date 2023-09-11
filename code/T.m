function x0=T(x,nfea,i,pro,DATA,k,ntrain)
x0=zeros(nfea+1,1);
    for j=1:k
        dj=[DATA{j},ones(ntrain(j),1)];
        if i~=j
            x0=x0+pro(i,j)*sum((sign(dj*x)).*dj)';
        end
    end
end