function [xh, rh, normrh, relresh, j, mvm_h, resvec_house, loss, lambda] = House(A, b, m, x0, r0, normr0, tol, mvm_h, resvec_house)
    
    %define variables
    [~,n] = size(A);
    V = eye(n,m); %orthonormal basis vectors
    HT = @(u,x) (x - 2*u*(u'*x)); %Householder transformation
    U = zeros(n,m); %Householder vectors
    R = zeros(m); %upper triangular matrix, st H=Q*R
    g = zeros(m+1,1); %last entry of Q*beta*e1 (approx. residual norm)
    c = zeros(1,m); %Givens coefficients
    s = zeros(1,m);
    
    %compute first Householder vector, such that
    % Pv = alpha * e1, with P = (I-2uu') & ||u|| = 1
    % u = (r0 - alpha * e1) / ||u||  and  alpha = -sgn(r0(1)) * ||r0||
    u = r0; 
    if (u(1) ~= 0)
        sgn = sign(u(1));
    else
        sgn = 1;
    end
    alpha = - sgn * norm(r0); 
    u(1) = u(1) - alpha; 
    u = u/norm(u);
    U(:,1) = u; %save Householder vector
    
    %save residual norm
    g(1) = alpha;

    %building m-dimensional Krylov subspace
    for j = 1:m
        
        %construct ONB {v1,...,vm} with Householder Arnoldi
        for k = j:-1:1
            V(:,j) = HT(U(:,k),V(:,j)); %apply previous HT: vj = P1...Pjej
        end
        V(:,j) = V(:,j)/norm(V(:,j)); %normalise

        %save loss of orthogonality
        loss(j) = norm(V(:,1:j)'*V(:,1:j)-eye(j), 'fro');
        
        %compute v = A*vj
        v = A*V(:,j);
        mvm_h = mvm_h+1; %matrix vector product with A

        %apply previous HT PjPj-1 ... P1v
        for k = 1:j
            v = HT(U(:,k),v); 
        end
        
        %construct next HT Pj+1
        if j ~= length(v)
            %compute next Householder vector: w = v(j+1:n,1), 
            %u = (w - alpha * e1) / ||u||  and  alpha = - sgn(v(1)) * ||w||
            u = v(j+1:n,1);
            if (u(1)~=0)
                sgn=sign(u(1));
            else
                sgn=1;
            end
            alpha = -sgn * norm(u);
            u(1) = u(1) - alpha;
            u = u/norm(u);
            U(j+1:n,j+1) = u;
            
            %apply Pj+1 to v -> jth column of Hessenberg matrix H
            v(j+2:end) = 0;
            v(j+1) = alpha;
            H(:,j) = v;
        end

        %apply previous Givens rotations to v
        for i = 1:j-1
            v(i:i+1)=  [conj(c(i)), conj(s(i)); - (s(i)), c(i)] * v(i:i+1);
        end

        %compute and apply next Givens rotation
        if j~=length(v)
            G = planerot(v(j:j+1));  %compute Givens rotation
            c(j) = G(2,2); %save Givens coefficients
            s(j) = -G(2,1);
            v(j:j+1) = G*v(j:j+1); %apply Givens rotation to H, st Q*H=R triangular
            g(j:j+1) = G*g(j:j+1); %apply Givens rotation to last column of Q
        end
        
        %save rotated vector in upper triangular R
        R(1:j,j) = v(1:j);
        
        %compute approximate relative residual norm
        relnormg = abs(g(j+1))/normr0;
        resvec_house(mvm_h) = relnormg; %save relative residual norm

        %Computation of Householder solution xh
        if (relnormg <= tol || j == m)
            yh = R(1:j, 1:j)\g(1:j); %divide upper triangular by g
            xh = x0 + V(:,1:j) * yh; %form solution x=x0+Vk*y 
            rh = b - A * xh; %compute residual
            normrh = norm(rh); %compute residual norm
            relresh = normrh/normr0; %compute relative residual norm
           
            %compute roots of residual polynomial (following Meurant)
            %H + h_ {n+1,n}^2 H^{-*}e_n e_n^T
            if m~=n
                e = eye(j);
                Hhat = H(1:j, 1:j) +H(j+1, j).^2 * inv(H(1:j, 1:j)')*e(:,j)*e(:,j)';
                [~,D] = eig(Hhat);
                lambda = diag(D); 
            end
        
            if relresh <= tol %relative residual sastisfies tolerance
                    break
            else
                    resvec_house(mvm_h) = relresh; %save relative residual norm
            end
        end %endif
    end %endfor
end