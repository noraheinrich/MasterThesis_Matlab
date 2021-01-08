function [xh, rh, normrh, relresh, j, mvm_h, resvec_house, loss, lambda] = WHouse(A, b, m, x0, r0, W, normr0, tol, mvm_h, resvec_house)
    
    %define variables
    [~,n] = size(A);
    V = eye(n,m); %orthonormal basis vectors
    HT = @(u,x) (x - 2*u*(u'*W*x)); %weighted Householder transformation
    U = zeros(n,m); %Householder vectors
    R = zeros(m); %upper triangular matrix, st H=Q*R
    g = zeros(m+1,1); %last entry of Q*beta*e1 (approx. W-residual norm)
    c = zeros(1,m); %Givens coefficients
    s = zeros(1,m);

    %sqrt of weights (needed for upper Hessenberg matrix)
    sqrtW = diag(sqrt(diag(W)));

    %compute first Householder vector, such that
    % Pv = alpha * e1, P = (I-2uu'W) & ||u||_W = 1
    % u = (r0 - alpha * e1) / ||u||  and  alpha = -sgn(r0(1)) * 1/sqrt(w1) * ||r0||_W
    u = r0; 
    if (u(1) ~= 0)
        sgn = sign(u(1));
    else
        sgn = 1;
    end
    normr = sqrt(r0' * W * r0); %weighted norm
    alpha = -sgn * 1/sqrt(W(1,1)) * normr;
    u(1) = u(1) - alpha;
    normu = sqrt(u' * W * u); %update weighted norm
    u = u/normu;
    U(:,1) = u; %save Householder vector
   
    %save residual norm
    g(1) = sqrtW(1,1) * alpha;
          
    %building m-dimensional Krylov subspace
    for j = 1:m
        
        %weighted unit vector 
        V(:,j) = 1/sqrt(W(j,j)) * V(:,j);
        
        %construct ONB {v1,...,vm} with Householder Arnoldi
        for k = j:-1:1
            V(:,j) = HT(U(:,k),V(:,j)); %apply previous HT: vj = P1...Pj*1/sqrt(w_j)*ej
        end
        V(:,j) = V(:,j)/sqrt(V(:,j)'*W*V(:,j)); %normalise in weighted norm
       
        %loss of orthogonality
        loss(j) = norm(V(:,1:j)'*W*V(:,1:j)-eye(j), 'fro'); 
       
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
            %u = (w - alpha * e1)/||u||  &  alpha = - sgn(v(1)) * 1/sqrt(wj+1) * ||w||
            u = v(j+1:n,1);
            if (u(1)~=0)
                sgn=sign(u(1));
            else
                sgn=1;
            end
            
            normu = sqrt(u' * W(j+1:n,j+1:n) * u); %weighted norm
            alpha = -sgn * 1/sqrt(W(j+1,j+1)) * normu; 
            u(1) = u(1) - alpha;
            normu = sqrt(u'*W(j+1:n,j+1:n)*u); %weighted norm
            u = u/normu;
            U(j+1:n,j+1) = u;
            
            %apply Pj+1 to v -> jth column of Hessenberg matrix H
            v(j+2:end) = 0;
            v(j+1) = alpha; 
            v = sqrtW * v; %retrieve Hessenberg matrix
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
        
        %computation of Householder solution xh and rel. Euclidean residual norm
        yh = R(1:j, 1:j)\g(1:j); %divide upper triangular by g
        xh = x0 + V(:,1:j) * yh; %form solution x=x0+Vk*y 
        rh = b - A*xh; %compute residual
        normrh = norm(rh); %compute Euclidean residual norm
        relresh = normrh/normr0; %compute relative Euclidean residual norm
        resvec_house(mvm_h) = relresh; %save relative Euclidean residual norm

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
        end
    end %endfor
end %endif