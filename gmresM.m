function [resvec_mgs, resvec_house, loss_m, loss_h] = gmresM(A, m, b, weight, tol, maxit) 
%restarted and weighted GMRES with with either MGS or Householder orthogonalization

%Input:
    %system matrix A
    %restart parameter m
    %right-hand side b
    %weighting strategy (euclidean - e, essai - w1, embree - w2, random - w3)
    %tolerance tol
    %maximum number of iterations
    %possible modification: orthogonalization strategy
    
%Output:
    %relative residual vectors
    %loss of orthogonality
    
%References: 
%Matlab GMRES implementation
%Algorithm 11.4.2 in "Matrix Computations" by Golub and van Loan
%Householder Arnoldi by Walker

%check conditions
%save and check size of A
[mm,n]=size(A);
if mm~=n
    error('System matrix not square')
end
%check restart parameter
if m > n
    warning('Restart parameter too large, set to size of A')
    m=n;
end 

%set defaults, if not specified
%random right-hand side
if (nargin < 3) 
    rng(1319);
    b=rand(n,1);
end
%no weighting strategy (i.e. restarted GMRES)
if (nargin < 4) 
    weight = 'e';
end
%tolerance
if (nargin < 5) || isempty(b) 
    tol = 10^-16;
end
%maximum number of (outer) iterations, i.e. cycles
if (nargin < 6) || isempty(tol) 
    maxit = 500;
end

%stagnation variables
stagmax = 5; %maximum number of stagnating cycles
stag_m = 0; %count of stagnating cycles
stag_h = 0; 

%Full GMRES: no restarts, i.e. only one outer iteration
if (m==0)
    m = n;
    maxit = 1;
end   

%initial approximation
x0 = zeros(n,1); 

%check all zero solution
normr0 = norm(b);
if normr0 <= tol             
    disp('zero vector already good enough')
    return
end


%START METHOD

%convergence flag is set to 1 (= did not converge)
flag_mgs = 1; 
flag_house = 1;

%initial approximations for MGS and Householder
xm0 = x0;
xh0 = x0;

%save first relative residual
mvm_m = 1;
mvm_h = 1;
resvec_mgs(mvm_m) = 1; 
resvec_house(mvm_h) = 1;

%initial weighting matrix set to identity matrix
if (weight ~= 'e')
    Wm = eye(n);
    Wh = eye(n);
end

%outer iterations (cycles)
for iter = 1:maxit

    %MGS GMRES
    if flag_mgs == 1
        
        rm0 = b - A*xm0; %initial residual
         
        %(W)MGS routine
        if weight == 'e'
            %no weighting: Euclidean inner product
            [xm, rm, normrm, relresm, inner, mvm_m, resvec_mgs, loss] = Mgs(A, b, m, xm0, rm0, normr0, tol, mvm_m, resvec_mgs); %compute basis and QR
        else
            %weighting: Weighted inner product
            [xm, rm, normrm, relresm, inner, mvm_m, resvec_mgs, loss] = WMgs(A, b, m, xm0, rm0, Wm, normr0, tol, mvm_m, resvec_mgs); %compute basis and QR
        end
        
        %save loss of orthogonality
        if m == n %full GMRES
            loss_m = loss; 
        else %restarted and weighted GMRES
            loss_m(iter) = loss(end); %save loss after each cycle
        end
        
        %check stagnation
        if (iter > 1 && abs(resvec_mgs(iter)-resvec_mgs(iter-1)) < eps)
            stag_m = stag_m + 1;
        elseif stag_m > 0
            stag_m = 0; %set counter back to zero
        end
            
        %check for status (converged, stagnated or prepare for next cycle)
        if relresm <= tol
            flag_mgs = 0; %converged
            if m == n %full GMRES
                fprintf('Full MGS-GMRES converged in iteration %d with a solution with a relative residual %d .\n',inner, relresm);
            elseif strcmp(weight,'e') %unweighted restarted GMRES
                fprintf('MGS-GMRES(%d) converged in outer iteration %d (inner iteration %d) with a solution with a relative residual %d .\n',m,iter,inner, relresm);
            else %weighted restarted GMRES
                fprintf('Weighted MGS-GMRES(%d) converged in outer iteration %d (inner iteration %d) with a solution with a relative residual %d .\n',m,iter,inner, relresm);
            end
        elseif stag_m == stagmax
                %flag_mgs = 3; %if needed to stop
                if m == n %full GMRES
                    fprintf('Full MGS-GMRES stagnated in outer iteration %d (inner iteration %d) with a solution with a relative residual %d .\n',iter,inner, relresm);
                elseif strcmp(weight,'e') %unweighted restarted GMRES
                    fprintf('MGS-GMRES(%d) stagnated in outer iteration %d (inner iteration %d) with a solution with a relative residual %d .\n',m,iter,inner, relresm);
                else %weighted restarted GMRES
                    fprintf('Weighted MGS-GMRES(%d) stagnated in outer iteration %d (inner iteration %d) with a solution with a relative residual %d .\n',m,iter,inner, relresm);
                end
        else
            xm0 = xm; %set next initial approximation
        end

        %Compute next weighting matrix W 
        if (~strcmp(weight,'e') && flag_mgs == 1)
            if strcmp(weight,'w1') %Essai 
               w = sqrt(n) * abs(rm)./normrm;    
            elseif strcmp(weight,'w2') %Embree et al.
                w = max(abs(rm)./norm(rm,inf), 10^-10);
            elseif strcmp(weight,'w3') %random
                if n <= 500
                    w = 0.5+rand(1,n)*(1.5-0.5);
                else
                    w = 2*rand(1,n);
                end
            end
            Wm = diag(w); %save next weighting matrix

            %check condition of weighting matrix
            if (~strcmp(weight,'e') && ismember(0, rcond(Wm)))
                flag_mgs = 3;
                'Weighting matrix is (close to) singular'
            end
        end

    end

    %Householder version
    if flag_house == 1

        rh0 = b - A*xh0; %initial residual

        %(W)House routine
        if weight == 'e'
            %no weighting: Euclidean inner product
            [xh, rh, normrh, relresh, inner, mvm_h, resvec_house, loss] = House(A, b, m, xh0, rh0, normr0, tol, mvm_h, resvec_house); %compute basis and QR
        else
            %weighting: Weighted inner product
            [xh, rh, normrh, relresh, inner, mvm_h, resvec_house, loss] = WHouse(A, b, m, xh0, rh0, Wh, normr0, tol, mvm_h, resvec_house); %compute basis and QR
        end
        
        %to save roots of residual polynomial, the additional 
        %output "lambda" of House() or WHouse() is needed
        %eigD(:,iter) = lambda; %save roots
        
        %loss of orthogonality
        if m == n %full GMRES
            loss_h = loss; %fuer m=n
        else
            loss_h(iter) = loss(end); %save loss after each cycle
        end
          
        %check stagnation
        if (iter > 1 && abs(resvec_house(iter)-resvec_house(iter-1)) < eps)
           stag_h = stag_h + 1;
        elseif stag_h > 0
            stag_h = 0; %set counter back to zero
        end          
       
        %check for status (converged, stagnated or prepare for next cycle)
        if relresh <= tol
            flag_house = 0; %converged
            if m == n %full GMRES
                fprintf('Full House-GMRES converged in iteration %d with a solution with a relative residual %d .\n', inner, relresh);
            elseif strcmp(weight,'e') %unweighted restarted GMRES
                fprintf('House-GMRES(%d) converged in outer iteration %d (inner iteration %d) with a solution with a relative residual %d .\n', m, iter,inner, relresh);
            else %weighted restarted GMRES
                fprintf('Weighted House-GMRES(%d) converged in outer iteration %d (inner iteration %d) with a solution with a relative residual %d .\n',m,iter,inner, relresh);
            end
        elseif stag_h == stagmax
                %flag_house = 3; %needed to stop
                if m == n %full GMRES
                    fprintf('Full House-GMRES stagnated in outer iteration %d (inner iteration %d) with a solution with a relative residual %d .\n', iter, inner, relresh);
                elseif strcmp(weight,'e') %unweighted restarted GMRES
                    fprintf('House-GMRES(%d) stagnated in outer iteration %d (inner iteration %d) with a solution with a relative residual %d .\n', m, iter, inner, relresh);
                else %weighted restarted GMRES
                    fprintf('Weighted House-GMRES(%d) stagnated in outer iteration %d (inner iteration %d) with a solution with a relative residual %d .\n', m, iter, inner, relresh);
                end    
        else
            xh0 = xh; %set next initial approximation
        end        

        %Compute next weighting matrix W 
        if (~strcmp(weight,'e') && flag_house == 1 )
            if strcmp(weight,'w1') %Essai 
               w = sqrt(n) * abs(rh)./normrh;    
            elseif strcmp(weight,'w2') %Embree et al
                w = max(abs(rh)./norm(rh,inf), 10^-10);
            elseif strcmp(weight,'w3') %random
                if n <= 500
                    w = 0.5+rand(1,n)*(1.5-0.5);
                else
                    w = 2*rand(1,n);
                end
            end
            Wh = diag(w); %save next weighting matrix

            %check condition of weighting matrix
            if (~strcmp(weight,'e') && ismembertol(0, rcond(Wh)))
                 flag_house = 3;
                 'Weighting matrix is (close to) singular'
            end
        end
        
    end


%end outer iteration: maxit reached
end 


%end function
end