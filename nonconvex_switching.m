% NONCONVEX_SWITCHING test script for nonconvex switching control
% This m-file solves the parabolic switching control problem
%  min 1/2 \|y-yd\|^2 + \alpha/2 \int_0^T |u(t)|_2^2 dt + \eta/2 \int_0^T
%  |u'(t)|_2^2 dt +\beta \int_0^T |u_1(t) u_2(t)| dt
%         + \gamma/2 \int_0^T(u_1(t)u_2(t))^2 dt
%      s.t. y_t - \Delta y = Bu, \partial_\nu y = 0, y(0) = 0
% using the approach described in the paper
%  "Nonconvex penalization of switching control of partial differential equations"
% by Christian Clason, Karl Kunisch, and Armin Rund,
% http://arxiv.org/abs/1605.09750.
%
% May 25, 2016                          Armin Rund <armin.rund@uni-graz.at>
%                            Christian Clason <christian.clason@uni-due.de>

% problem statement
d.example = 2;                  % example 1 or 2 from the paper (2 is default)
d.alpha = 1e-6;                 % quadratic penalty
d.eta = 1e-5;                   % H1-regularization
d.be_min = -5;                  % minimal value (L1 switching)
d.ga_min = -9;                  % minimal value (L2 switching)
d.ga_max = 4;                   % maximal value (L2 switching)

% discretization:
d.h = 0.1;                      % mesh size parameter hmax
d.Nt = 101;                     % number of time points
d.Nc = d.Nt;                    % number of degrees of freedom for control
d.Tend = 10;                    % terminal time
d.t = linspace(0,d.Tend,d.Nt);  % time discretization
d.tau = d.t(2)-d.t(1);          % time step size (equidistant grid)

% parameters of the optimization method:
d.maxit_ssn  = 5;         % maximal iterations for SSN (semismooth Newton method)
d.abstol_ssn = 1e-7;      % absolute tolerance of SSN
d.reltol_ssn = 1e-6;      % relative tolerance of SSN
d.maxit_gmres   = 3*d.Nt; % maximal iterations for GMRES method
d.reltol_gmres  = 1e-8;   % relative tolerance of GMRES

% assemble finite element matrices:
[p,e,t] = initmesh('squareg','Hmax',d.h);
[d.A,d.M,~]  = assema(p,t,1,1,0);   % stiffness matrix A and mass matrix M
d.Mobs = d.M;                       % mass matrix Mobs of observation domain
d.Nx = size(d.M,1);                 % number of discrete points in space
d.MpA = d.M + 0.5*d.tau*d.A;        % system matrices of CG(1)DG(0) Crank-Nicolson
d.MmA = d.M - 0.5*d.tau*d.A;

% mass and Laplace matrix for H1-regularization for cont. pw. linear functions:
et = ones(d.Nc,1);
d.At = spdiags([-et 2*et -et]/d.tau,-1:1,d.Nc,d.Nc); 
d.At(1,1) = 1/d.tau;    d.At(end,end) = 1/d.tau;      % homogeneous Neumann
d.Mt = spdiags([et 4*et et],-1:1,d.Nc,d.Nc);
d.Mt(1,1) = 2;     d.Mt(end,end) = 2;
d.Mt = d.Mt*d.tau/6;
d.Ml = diag(sum(d.Mt));   % lumped mass matrix

% assemble control operator and desired state vector
d.Bu = [];
d.yd = zeros(d.Nx,d.Nt);
if d.example < 2
    obs = 'x.^2 + y.^2 <= 0.5^2';   % observation domain
    [~,d.Mobs,~] = assema(p,t,0,obs,0); % mass matrix Mobs of observation domain
    for n = 1:2
        phi = pi/4 + (n-1)*2*pi/2;
        x = cos(phi)/sqrt(2);   y = sin(phi)/sqrt(2);
        str_fn = strcat('(x-(',num2str(x),')).^2 + (y-(',num2str(y),')).^2');
        [~,~,fn] = assema(p,t,0,0,strcat(str_fn,'<= 0.1^2'));
        d.Bu = [d.Bu,fn];
        
        xi = cos(n+d.t);
        [~,~,g] = assema(p,t,0,0,str_fn);
        g = d.M\g;
        h = xi.*sin(2*pi*d.t /d.Tend).^2;
        d.yd = d.yd + g*h;
    end
else
    [~,~,f1] = assema(p,t,0,0,'x<=0'); % left half
    [~,~,f2] = assema(p,t,0,0,'x>0');  % right half
    d.Bu = 0.1*[f1,f2];
    [~,~,g1] = assema(p,t,0,0,'sin(pi*x)'); % right lower part
    g1 = d.M\g1;
    h1 = sin(3*pi*d.t /d.Tend);
    d.yd = g1*h1;
    u1 = 20*sin(2*pi*d.t(1:d.Nc)/d.Tend).^4';
    u2 = 10*cos(0.7*2*pi*d.t(1:d.Nc)/d.Tend).^4';
    u0 = [u1;u2;0*u1];
    d.beta = 0;    d.gamma = 0;
    [~,~,~,~,d.yd] = objfun(u0,d);
end
fprintf('starting example %d with Nt = %d    h = %.2f   alpha=%1.1e  eta=%1.1e  \n', ...
    d.example,d.Nt,d.h,d.alpha,d.eta);


%% compute control
u = zeros(2*d.Nc,1); % zero initial guess
v = [u; 0*ones(d.Nc,1)];
d.beta = 10^d.be_min;
for k = d.ga_min:d.ga_max  % homotopy in gamma (L2-switching)
    d.gamma = 10^k;
    % semi-smooth Newton method working on v = [u,q]
    [v,output] = ssn(v,d);
end
for k = d.be_min:min(d.be_min+20,10) % homotopy in beta (L1-switching)
    d.beta = 10^k;
    [v,output] = ssn(v,d);
    quality = max(v(1:d.Nc).*v(d.Nc+1:2*d.Nc)); maxu = max(abs(v(1:2*d.Nc)));
    fprintf('   err_sw = %1.2e   max|u|=%1.2e   err_sw/max|u|)=%1.2e\n',quality,maxu,quality/maxu);
    if quality/maxu < 1e-10 , break; end;
end
bemax = d.beta;
for k = d.ga_max-1:-1:d.ga_min  % homotopy in gamma
    d.gamma = 10^k;
    [v,output] = ssn(v,d);
end

%% postprocessing:
u1 = v(1:d.Nc); u2 = v(d.Nc+1:2*d.Nc); q = v(2*d.Nc+1:end);
switches = 0 ;% count the number of switches:
for n=1:d.Nc-1
    if abs(u1(n))>=abs(u2(n)) && abs(u1(n+1))<abs(u2(n+1))
        switches = switches + 1;
    end
    if abs(u1(n))<abs(u2(n)) && abs(u1(n+1))>=abs(u2(n+1))
        switches = switches + 1;
    end
end
fprintf('  number of switches =  %3d   and  beta_max = %1.1e \n',switches,bemax);

% plot control
t = linspace(0,d.Tend,d.Nc);
up = reshape(u(1:2*d.Nc),d.Nc,2);
plot(t,up,'-','LineWidth',1.5); set(gca,'FontSize',20    );
title(['optimal control for \alpha = ',num2str(d.alpha,'%1.0e'),' \eta = ', num2str(d.eta,'%1.0e')]);
