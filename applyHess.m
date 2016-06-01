function Hdv = applyHess(dv,v,d)
% APPLYHESS compute application of Hessian
% HDV = APPLYHESS(DV,V,D) computes the action HDV of the Hessian in
% direction DV. V is the current point. The structure D contains the problem parameters.
%
% May 25, 2016                          Armin Rund <armin.rund@uni-graz.at>
%                            Christian Clason <christian.clason@uni-due.de>

% point and direction, separated in control u and subdifferential element q
u = v(1:d.Nc*2);  u1 = u(1:d.Nc); u2 = u(d.Nc+1:2*d.Nc);
q = v(d.Nc*2+1:end);
du = dv(1:d.Nc*2);  du1 = du(1:d.Nc); du2 = du(d.Nc+1:2*d.Nc);
dq = dv(2*d.Nc+1:end);

% solve linearized state equation
dy = zeros(d.Nx,d.Nt);        % zero initial condition
for m = 1:d.Nt-1              % time stepping
    dy(:,m+1) = d.MpA\(d.MmA*dy(:,m) + 0.5*d.tau*d.Bu*(du(m:d.Nc:end)+du(m+1:d.Nc:end)));
end

% solve linearized adjoint equation for dz
BTdz = 0*du;                   % initialization of B'*dz
dz = d.MpA\(0.5*d.tau*d.Mobs*dy(:,d.Nt));  % terminal condition
BTdz(d.Nc:d.Nc:end) = d.Bu'*dz;
for m = d.Nt-1:-1:2            % time stepping (backward)
    dz = d.MpA\(d.MmA*dz + d.tau*d.Mobs*dy(:,m));
    BTdz(m:d.Nc:end) = d.Bu'*dz;
end
BTdz = 0.5*(BTdz+[BTdz(2:d.Nc);0;BTdz(d.Nc+2:2*d.Nc);0]);

% Hessian direction
Hdv = [([d.tau*BTdz(1:d.Nc);d.tau*BTdz(d.Nc+1:2*d.Nc)] + d.alpha*[d.Ml*du1;d.Ml*du2]...
    + d.eta*[d.At*du1;d.At*du2] +[d.Ml*(q.*du2);d.Ml*(q.*du1)] +[d.Ml*(dq.*u2);d.Ml*(dq.*u1)]); ...
    d.Ml*(d.gamma.*du1.*u2 +d.gamma.*u1.*du2 -dq.*(abs(q)>=d.beta))];
