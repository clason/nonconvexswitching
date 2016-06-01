function [obj,grad,quality,active,y,tracking] = objfun(v,d)
% OBJFUN compute functional value, gradient
% [OBJ,GRAD,QUALITY,ACTIVE,Y,TRACKING] = OBJFUN(V,D) computes the value OBJ of the
% functional to be minimized together with the gradient GRAD in the point
% V=(U,Q). QUALITY is the switching quality for output, defined as the
% maximal value of abs(u1*u2) over time. ACTIVE is a time vector containing 
% the index of the defacto acting control (the control component with the 
% largest absolute at that time).
% Other outputs are the state Y and the tracking quality TRACKING.
% The structure D contains the problem parameters.
%
% May 25, 2016                          Armin Rund <armin.rund@uni-graz.at>
%                            Christian Clason <christian.clason@uni-due.de>

u = v(1:d.Nc*2);  u1 = u(1:d.Nc);  u2 = u(d.Nc+1:2*d.Nc);
q = v(d.Nc*2+1:end); % auxilary variables

% solve state equation
y = zeros(d.Nx,d.Nt);         % zero initial condition
for m = 1:d.Nt-1              % time stepping
    y(:,m+1) = d.MpA\(d.MmA*y(:,m) + 0.5*d.tau*d.Bu*(u(m:d.Nc:end)+u(m+1:d.Nc:end)));
end
res = y - d.yd;               % residual

% summed trapezoidal rule for the tracking term
normres  = diag(res'*d.Mobs*res);
tracking = 0.5*d.tau*(sum(normres) - 0.5*(normres(1)+normres(end)));

L2_costs = 0.5*(u1'*d.Ml*u1 + u2'*d.Ml*u2);
H1_costs = 0.5*(u1'*d.At*u1 + u2'*d.At*u2);  % objective part for H1-regularization
L2_sw = 0.5*(u1.^2)'*d.Ml*(u2.^2);
L1_sw = abs(u1)'*d.Ml*abs(u2);
obj = tracking + d.alpha*L2_costs + d.eta*H1_costs + d.beta*L1_sw +d.gamma*L2_sw;

%% compute gradient
BTz = 0*u;                                   % initialization of B'*z

% solve adjoint equation for adjoint state z
z = d.MpA\(0.5*d.tau*d.Mobs*res(:,d.Nt));  % terminal condition
BTz(d.Nc:d.Nc:end) = d.Bu'*z;
for m = d.Nt-1:-1:2                          % time stepping (backward)
    z = d.MpA\(d.MmA*z + d.tau*d.Mobs*res(:,m));
    BTz(m:d.Nc:end) = d.Bu'*z;
end
BTz = 0.5*(BTz+[BTz(2:d.Nc);0;BTz(d.Nc+2:2*d.Nc);0]); % exact discrete derivatives for CnV

quality = max(u1.*u2);
active  = ones(d.Nc,1) + (abs(u2)>abs(u1)); % index of defacto acting control (largest control component)

grad = [[d.tau*BTz(1:d.Nc); d.tau*BTz(d.Nc+1:2*d.Nc)] + d.alpha*[d.Ml*u1;d.Ml*u2] ...
    + d.eta*[d.At*u1;d.At*u2] + [d.Ml*(q.*u2);d.Ml*(q.*u1)]; ...
    d.Ml*(d.gamma.*u1.*u2 -(q-d.beta).*(q>=d.beta) -(q+d.beta).*(q<=-d.beta))];
