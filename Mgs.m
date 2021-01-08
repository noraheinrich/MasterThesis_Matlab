function [xm, rm, normrm, relresm, j, mvm_m, resvec_mgs, loss] = Mgs(A, b, m, x0, r0, normr0,tol, mvm_m, resvec_mgs)
    
    %define variables
    [~,n] = size(A);
    V = zeros(n,m+1); %orthonormal basis vectors
    H = zeros(m+1,m); %upper Hessenberg
    R = zeros(m); %upper triangular matrix, st H=Q*R
    g = zeros(m+1,1); %last entry of Q*beta*e1 (approx. residual norm)
    c = zeros(1,m); %Givens coefficients
    s = zeros(1,m);

    %save first basis vector and residual norm
    normr = norm(r0);
    g(1) = normr;
    V(:,1) = r0/normr;
    
    %building m-dimensional Krylov subspace
    for j = 1:m
        
        %construct ONB {v1,...,vm} with MGS Arnoldi
        vhat = A * V(:,j); %next basis vector
        mvm_m = mvm_m+1; %matrix vector product with A
        for i = 1:j    
            R(i,j) = V(:,i)' * vhat; %orthogonalising
            vhat = vhat - R(i,j) * V(:,i);
        end
        R(j+1,j) = norm(vhat);
        V(:,j+1) = vhat/R(j+1,j); %save basis vector
        H(1:j+1,j) = R(1:j+1,j); %save upper Hessenberg
        
        %save loss of orthogonality
        loss(j) = norm(V(:,1:j)' * V(:,1:j)-eye(j), 'fro'); 
        
        %apply previous Givens rotations to H (for j>1)
        for i = 1:j-1 
            R(i:i+1,j) = [conj(c(i)),conj(s(i)); -s(i), c(i)] * R(i:i+1,j);
        end
        
        %compute and apply next Givens rotation
        G = planerot(R(j:j+1,j)); %compute Givens rotation
        c(j) = G(2,2); %save Givens coefficients
        s(j) = -G(2,1);
        R(j:j+1,j) = G * R(j:j+1,j); %apply Givens rotation to H, st Q*H=R triangular
        R(j+1,j) = 0; %against rounding errors
        g(j:j+1) = G*g(j:j+1); %apply Givens rotation to last column of Q

        %compute approximate relative residual norm 
        relnormg = abs(g(j+1))/normr0;
        resvec_mgs(mvm_m) = relnormg; %save relative residual norm

        %Computation of MGS solution xm
        if (relnormg <= tol || j == m)
            ym = R(1:j, 1:j)\g(1:j); %divide upper triangular by g
            xm = x0 + V(:,1:j) * ym; %form solution x=x0+Vk*y 
            rm = b - A*xm; %compute residual
            normrm = norm(rm); %compute residual norm
            relresm = normrm/normr0; %compute relative residual norm
            
            if relresm <= tol %relative residual sastisfies tolerance
                break
            else
                resvec_mgs(mvm_m) = relresm; %save relative residual norm
            end
        end %endif
    end %endfor
end