function [v,output] = ssn(v,d)
% SSN semismooth Newton method
% [V,OUTPUT] = SSN(V,D) computes the optimal control V from a given
% initial point V using a semismooth Newton method. The structure D contains
% the problem parameters, while OUTPUT is a structure containing the
% following data:
%     j:        optimal value
%     g0:       residual norm in optimality conditions
%     ssnit:    number of Newton steps
%     gmresit:  number of gmres steps in last Newton iteration
%     flag:  0 - converged with relative tolerance, 1 - converged with
%            absolute tolerance, 2 - diverged (too many iterations)
%
% May 25, 2016                          Armin Rund <armin.rund@uni-graz.at>
%                            Christian Clason <christian.clason@uni-due.de>

fprintf('Starting SSN for beta= %1.0e, gamma=%1.0e:      (Krylov flag 0: converged, 1: max iterations)\n',d.beta,d.gamma);
fprintf('Iter   objective  sw_quality | normgrad  | stepsize flag relres GMRESit | change in AS\n');
warning('off','MATLAB:gmres:tooSmallTolerance')

%% semismooth Newton iteration

ssnit = 0;  GGold = 1e99;  tau = 1; 
active_old = 0*v;  change = 0;
while ssnit <= d.maxit_ssn
    % compute new gradient
    [j,G,quality,active] = objfun(v,d);
    if ssnit == 0
        G0 = norm(G);  output.g0 = G0;  output.j0 = j;
        flag = 0;  gmresit = 0;
    end
    
    % line search on gradient norm (correctly scaled discrete norm)
    GG = norm(G);
    crit = (GG>=GGold);
    
    if crit % if no decrease: backtrack (never on iteration 1)
        tau = tau/2;
        v = v - tau*dv;
        if tau < 1e-7  % if step too small: terminate Newton iteration
            fprintf('\n#### not converged: step size too small\n');
            output.flag = 3;
            break;
        else             % else: bypass rest of loop; backtrack further
            continue;
        end
    end
    
    % output iteration details
    fprintf('%3d:  %1.5e    %1.0e   | %1.3e |', ...
        ssnit, j,  quality, GG);
    if ssnit > 0
        fprintf(' %1.1e   %d  %1.1e   %2d  |   %2d  %d\n', tau, flag, relres, floor(gmresit(end)),sum(active_old~=active),change);
    else
        fprintf('\n');
    end
    
    % terminate Newton?
    if (GG < d.reltol_ssn*sqrt(G0))   % convergence (relative norm)
        fprintf('\n#### converged with relative tol: |grad|<=%1.1e |grad0|\n',d.reltol_ssn);
        output.flag = 0;
        break;
    elseif (GG < d.abstol_ssn)  % convergence (absolute norm)
        fprintf('\n#### converged with absolute tol: |grad|<=%1.1e\n',d.abstol_ssn);
        output.flag = 1;
        break;
    elseif ssnit == d.maxit_ssn                     % failure, too many iterations
        fprintf('\n#### not converged: too many iterations\n');
        output.flag = 2;
        break;
    end
    % otherwise update information, continue
    ssnit = ssnit+1;  GGold = GG;  tau = 1;
    
    % compute Newton step, update
    DG = @(dv) applyHess(dv,v,d);  % Hessian
    [dv, flag, relres, gmresit] = gmres(DG,-G,[],d.reltol_gmres,d.maxit_gmres); % no restart
    v = v + dv;
    active_old=active;
end

%% output

output.j     = j;
output.g     = GG;
output.ssnit = ssnit;
output.gmresit  = gmresit;
output.switching = quality;

